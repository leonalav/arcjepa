import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import torch.distributed as dist
from pathlib import Path
import argparse
import sys

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    if hasattr(torch, '_register_device_module'):
        torch._register_device_module('xla', torch_xla)
    else:
        torch.xla = torch_xla
    IS_TPU = True
except ImportError:
    IS_TPU = False

torch.multiprocessing.set_sharing_strategy('file_system')
# Detect project root dynamically
PROJECT_ROOT = Path(__file__).parent.resolve()

# Ensure project root is in path for robust 'src.xxx' imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.world_model import ARCJEPAWorldModel
from src.training.loss import ARCJPELoss
from src.training.ema import EMAUpdater
from src.data.dataset import ARCTrajectoryDataset, create_mock_trajectory, FastHFARCDataset
from src.training.metrics import (
    compute_latent_metrics,
    compute_prediction_metrics,
    compute_gradient_metrics,
    compute_data_statistics
)

def setup_dist(local_rank_arg=-1):
    """Dynamically setup distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Multi-node / Multi-GPU (torchrun / deepspeed)
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # DeepSpeed launcher passes --local_rank, torchrun sets LOCAL_RANK env
        if local_rank_arg != -1:
            local_rank = local_rank_arg
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        # Single GPU / CPU
        return 0, 1, 0

def check_training_files_exist(data_path: Path) -> bool:
    """
    Look into the folder and check the existence of training files.
    If files exist (either in root or train/ split), return True to skip processing.
    """
    try:
        next(data_path.rglob("*.jsonl"))
        return True
    except StopIteration:
        return False

def train():
    parser = argparse.ArgumentParser(description="ARC-JEPA Training Loop")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context_ratio", type=float, default=0.7)
    parser.add_argument("--use_curriculum", action="store_true", help="Use curriculum learning for context ratio")
    parser.add_argument("--curriculum_start", type=float, default=0.9, help="Starting context ratio for curriculum")
    parser.add_argument("--curriculum_end", type=float, default=0.5, help="Ending context ratio for curriculum")
    parser.add_argument("--hf_repo_id", type=str, default=None, help="HuggingFace repo ID for FastHFARCDataset")
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data" / "recordings"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--recon_weight", type=float, default=0.01)
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed for training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    # Research extension flags
    parser.add_argument("--use_vicreg", action="store_true", help="Enable VICReg covariance loss")
    parser.add_argument("--vicreg_weight", type=float, default=25.0, help="VICReg covariance weight on raw latents")
    parser.add_argument("--var_weight", type=float, default=25.0, help="VICReg variance weight (separate from cov_weight)")
    parser.add_argument("--proj_cov_weight", type=float, default=5.0, help="VICReg covariance weight on projector outputs (keeps projector spanning full 1024D)")
    parser.add_argument("--multistep_k", type=int, default=1, help="Number of steps for multi-step prediction")
    parser.add_argument("--use_focal", action="store_true", help="Enable focal loss for reconstruction")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal loss alpha parameter")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--compute_temporal_masks", action="store_true", help="Compute temporal masks for focal loss")
    parser.add_argument("--temporal_weight", type=float, default=10.0, help="Weight multiplier for changed pixels")
    parser.add_argument("--filter_noops", action="store_true", help="Filter no-op trajectories from dataset")
    parser.add_argument("--num_trajectories", type=int, default=1000, help="Number of trajectories to generate")
    parser.add_argument("--logger", type=str, default="csv", choices=["wandb", "tensorboard", "csv"], help="Logging backend")

    args = parser.parse_args()

    # Tailor config for TPU v5e-8 (16GB HBM) to maximize throughput securely
    if IS_TPU and args.batch_size > 4:
        print(f"TPU detected: Tailoring batch_size from {args.batch_size} to 4 to prevent HBM OOM while maximizing throughput.")
        args.batch_size = 4

    # gdntpu (pure-PyTorch) is used on any non-CUDA path (TPU or CPU).
    # It requires strictly static sequence lengths so XLA can trace and cache
    # a single compiled graph.  We must never feed a partial final batch.
    use_gdntpu = IS_TPU or not torch.cuda.is_available()

    if IS_TPU:
        from accelerate import Accelerator
        accelerator = Accelerator()
        device = accelerator.device
        is_main_process = accelerator.is_main_process
        world_size = accelerator.num_processes
        local_rank = accelerator.local_process_index
        rank = accelerator.process_index
    else:
        accelerator = None
        # Initialize Distributed environment
        rank, world_size, local_rank = setup_dist(args.local_rank)
        is_main_process = (rank == 0)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 1. Setup Data
    if args.hf_repo_id is not None:
        if is_main_process:
            print(f"🚀 Using FastHFARCDataset to stream from HuggingFace repo: {args.hf_repo_id}")
        dataset = FastHFARCDataset(
            hf_repo_id=args.hf_repo_id, 
            split="train", 
            compute_temporal_masks=args.compute_temporal_masks
        )
    else:
        data_path = Path(args.data_dir)
        if is_main_process:
            data_path.mkdir(parents=True, exist_ok=True)
        
        # Wait for directory creation if multi-gpu
        if IS_TPU:
            accelerator.wait_for_everyone()
        elif world_size > 1: dist.barrier()
        
        # Check if files exist and generate if necessary (Main process only)
        if is_main_process:
            if not check_training_files_exist(data_path):
                print(f"No recordings found in {args.data_dir} (or its subdirectories). Generating real trajectories via ARC-AGI...")
                create_mock_trajectory(args.data_dir, num_trajectories=args.num_trajectories)
            else:
                print(f"Training files found in {args.data_dir}. Skipping data generation.")
            
        # ALL processes wait for generation to finish
        if IS_TPU:
            accelerator.wait_for_everyone()
        elif world_size > 1: dist.barrier()
        
        # CRITICAL: ALL processes re-read the directory after potential generation
        # Now strictly pointing to the 'train' split to prevent data contamination
        train_data_path = data_path / "train"
        recording_files = list(train_data_path.glob("*.jsonl"))
        
        if not recording_files:
            print(f"Warning: No recordings found in {train_data_path}. Ensure generation completed.")
            # Fallback to direct path if the user didn't use the split generator
            recording_files = list(data_path.glob("*.jsonl"))
            
        if not recording_files:
            if is_main_process: print("Error: No .jsonl files found.")
            sys.exit(1)

        # IMPORTANT — XLA static-shape requirement:
        # window_size must be a multiple of the GDN chunk_size (64) so that the
        # padded sequence length never changes between batches.  A window of 64
        # means every batch has shape [B, 64, H, W] — a single static XLA graph.
        gdn_chunk_size = 64
        window_size = gdn_chunk_size  # 64 steps per trajectory window
        dataset = ARCTrajectoryDataset(
            [str(f) for f in recording_files],
            window_size=window_size,
            multistep_k=args.multistep_k,
            compute_temporal_masks=args.compute_temporal_masks,
            filter_noops=args.filter_noops
        )

        # Validate ARC compliance
        if is_main_process:
            dataset.validate_arc_compliance()

    if IS_TPU:
        sampler = None
    else:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    # Use num_workers=0 for distributed to avoid potential multiproc deadlocks on some systems
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=12,
        pin_memory=not IS_TPU,
        # drop_last MUST be True on any path that uses gdntpu (TPU or CPU).
        # XLA traces one compiled graph per unique input shape.  A partial
        # final batch would create a second distinct graph, doubling compile
        # time and potentially causing out-of-memory during the second trace.
        drop_last=use_gdntpu,
    )

    # 2. Initialize Model & Components
    model = ARCJEPAWorldModel(
        d_model=512,  # Increased from 256
        num_vit_layers=6,  # Increased from 4
        num_gdn_heads=8,  # Increased from 4
        multistep_k=args.multistep_k
    )
    if not IS_TPU:
        model = model.to(device)
    
    # DeepSpeed Integration (Optional)
    if not IS_TPU and args.deepspeed:
        try:
            import deepspeed
            # DeepSpeed expects local_rank in its init
            ds_config = {
                "train_batch_size": args.batch_size * world_size,
                "optimizer": {"type": "AdamW", "params": {"lr": args.lr}},
                "fp16": {"enabled": True},
                "zero_optimization": {"stage": 2}
            }
            model_engine, optimizer, _, _ = deepspeed.initialize(
                args=args, model=model, model_parameters=model.parameters(), config=ds_config
            )
            model = model_engine
        except ImportError:
            if is_main_process: print("DeepSpeed not found, falling back to standard DDP.")
            args.deepspeed = False

    # Standard DDP if multi-gpu and not using DeepSpeed
    if not IS_TPU and world_size > 1 and not args.deepspeed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    criterion = ARCJPELoss(
        recon_weight=args.recon_weight,
        use_vicreg=args.use_vicreg,
        vicreg_weight=args.vicreg_weight,
        var_weight=args.var_weight,
        proj_cov_weight=args.proj_cov_weight,
        use_focal=args.use_focal,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        temporal_weight_multiplier=args.temporal_weight
    )
    if IS_TPU or not args.deepspeed:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        
    if IS_TPU:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # EMA Updater
    if IS_TPU:
        unwrapped = accelerator.unwrap_model(model)
        online_enc = unwrapped.online_encoder
        target_enc = unwrapped.target_encoder
    else:
        online_enc = model.module.online_encoder if hasattr(model, 'module') else model.online_encoder
        target_enc = model.module.target_encoder if hasattr(model, 'module') else model.target_encoder
        
    # I-JEPA paper: start tau=0.996, linearly anneal to 1.0 throughout training.
    # This creates a stronger learning signal early (target diverges more from online)
    # and gradually stabilises as training converges.
    ema_updater = EMAUpdater(online_enc, target_enc, tau_start=0.996, tau_end=1.0)

    # Initialize logger
    logger = None
    if is_main_process:
        if args.logger == "wandb":
            try:
                import wandb
                wandb.init(project="arc-jepa", config=vars(args))
                logger = wandb
            except ImportError:
                print("wandb not installed, falling back to CSV logging")
                args.logger = "csv"
        elif args.logger == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                logger = SummaryWriter(log_dir=str(Path(args.output_dir) / "logs"))
            except ImportError:
                print("tensorboard not installed, falling back to CSV logging")
                args.logger = "csv"

        if args.logger == "csv":
            # CSV logger will be handled manually
            csv_path = Path(args.output_dir) / "metrics.csv"
            csv_path.parent.mkdir(exist_ok=True)
            logger = None

    # 3. Training Loop
    model.train()
    if is_main_process:
        print(f"Starting training on {world_size} GPU(s) using {'DeepSpeed' if args.deepspeed else 'DDP'}...")
        print(f"Dataset size: {len(dataset)}")
        print(f"Extensions: VICReg={args.use_vicreg}, MultiStep={args.multistep_k}, Focal={args.use_focal}")
        if use_gdntpu:
            print(
                "Note: First forward+backward pass will take several minutes while "
                "XLA traces and compiles the GDN graph (pure_chunk_gated_delta_rule "
                "nested loops + all surrounding ops). Every subsequent batch will be "
                "lightning fast — the compiled graph is cached for the entire run."
            )
        else:
            print("Note: First batch may take ~30 s for Triton kernel JIT compilation.")

    # CSV logging setup
    csv_metrics = []

    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        total_loss = 0

        # I-JEPA EMA annealing: linearly increase tau from tau_start to tau_end.
        # progress=0.0 at epoch 0 → tau=tau_start (0.996)
        # progress=1.0 at last epoch → tau=tau_end (1.0)
        ema_progress = epoch / max(1, args.epochs - 1)
        ema_updater.set_progress(ema_progress)

        # Curriculum learning for context ratio
        if args.use_curriculum:
            progress = epoch / args.epochs
            context_ratio = args.curriculum_start - progress * (args.curriculum_start - args.curriculum_end)
            context_ratio = max(args.curriculum_end, context_ratio)
        else:
            context_ratio = args.context_ratio

        for batch_idx, batch in enumerate(dataloader):
            if is_main_process and batch_idx == 0:
                print(f"Processing first batch of epoch {epoch+1}...")
                if args.use_curriculum:
                    print(f"  Context ratio: {context_ratio:.2f}")

            if not IS_TPU:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if args.deepspeed and not IS_TPU:
                outputs = model(batch, context_ratio=context_ratio, use_teacher_forcing=True)
                loss_dict = criterion(outputs, batch)
                loss = loss_dict['loss']
                model.backward(loss)
                model.step()
            elif IS_TPU:
                optimizer.zero_grad()
                outputs = model(batch, context_ratio=context_ratio, use_teacher_forcing=True)
                loss_dict = criterion(outputs, batch)
                loss = loss_dict['loss']
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                optimizer.zero_grad()
                outputs = model(batch, context_ratio=context_ratio, use_teacher_forcing=True)
                loss_dict = criterion(outputs, batch)
                loss = loss_dict['loss']
                loss.backward()
                optimizer.step()

            ema_updater.update()
            total_loss += loss.detach()

            # Compute comprehensive metrics (every 10 batches)
            # CRITICAL: Move all metric computation to CPU after a single sync
            # to prevent graph breaks from .item(), eigvalsh, histc, etc.
            if batch_idx % 10 == 0:
                # Single sync point — flush the XLA graph
                if IS_TPU:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()

                with torch.no_grad():
                    # Transfer to CPU ONCE, then compute metrics without graph breaks
                    target_lat_cpu = outputs['target_latents'].detach().cpu()
                    logits_cpu = outputs['decoder_logits'].detach().cpu()
                    final_state_cpu = batch['final_state'].detach().cpu() if isinstance(batch['final_state'], torch.Tensor) else batch['final_state']
                    states_cpu = batch['states'].detach().cpu()
                    target_states_cpu = batch['target_states'].detach().cpu()
                    temporal_mask_cpu = batch.get('temporal_mask', None)
                    if temporal_mask_cpu is not None and isinstance(temporal_mask_cpu, torch.Tensor):
                        temporal_mask_cpu = temporal_mask_cpu.detach().cpu()

                    seq_mask_cpu = batch.get('seq_mask', None)
                    if seq_mask_cpu is not None and isinstance(seq_mask_cpu, torch.Tensor):
                        seq_mask_cpu = seq_mask_cpu.detach().cpu()

                    latent_metrics = compute_latent_metrics(target_lat_cpu, seq_mask=seq_mask_cpu)
                    pred_metrics = compute_prediction_metrics(
                        logits_cpu,
                        final_state_cpu,
                        temporal_mask_cpu,
                        states_cpu
                    )
                    grad_metrics = compute_gradient_metrics(model)
                    data_metrics = compute_data_statistics(states_cpu, target_states_cpu, seq_mask=seq_mask_cpu)

                    # Materialize loss_dict values to Python floats for logging
                    loss_dict_cpu = {}
                    for k_name, v_val in loss_dict.items():
                        if isinstance(v_val, torch.Tensor):
                            loss_dict_cpu[k_name] = v_val.item()
                        else:
                            loss_dict_cpu[k_name] = v_val

                    # Aggregate all metrics
                    metrics = {
                        'epoch': epoch + 1,
                        'batch': batch_idx,
                        'lr': optimizer.param_groups[0]['lr'] if not args.deepspeed else args.lr,
                        **loss_dict_cpu,
                        **latent_metrics,
                        **pred_metrics,
                        **grad_metrics,
                        **data_metrics
                    }

                    # Log to backend
                    if is_main_process:
                        if args.logger == "wandb" and logger is not None:
                            logger.log(metrics)
                        elif args.logger == "tensorboard" and logger is not None:
                            for key, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    logger.add_scalar(key, value, epoch * len(dataloader) + batch_idx)
                        else:
                            # CSV logging
                            csv_metrics.append(metrics)

                        # Print critical metrics
                        print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}]")
                        print(f"  Loss: {loss_dict_cpu['loss']:.4f} | JEPA: {loss_dict_cpu['jepa_loss']:.4f} | Recon: {loss_dict_cpu['recon_loss']:.4f} | Std: {loss_dict_cpu['std_loss']:.4f} | Policy: {loss_dict_cpu['policy_loss']:.4f}")
                        print(f"  Latent: std={latent_metrics['latent_std_mean']:.3f} rank={latent_metrics['effective_rank']:.1f} corr={latent_metrics['latent_correlation_max']:.3f}")
                        print(f"  Accuracy: pixel={pred_metrics['pixel_accuracy']:.3f} fg={pred_metrics['foreground_accuracy']:.3f} bg={pred_metrics['background_accuracy']:.3f}")
                        if args.compute_temporal_masks:
                            print(f"  Temporal: changed={pred_metrics['changed_pixel_accuracy']:.3f} unchanged={pred_metrics['unchanged_pixel_accuracy']:.3f}")
                        print(f"  Data: noop={data_metrics['noop_ratio']:.3f} fg_ratio={data_metrics['foreground_ratio']:.3f}")
                        if args.use_vicreg:
                            print(f"  VICReg: cov_raw={loss_dict_cpu['cov_loss_raw']:.4f} cov_proj={loss_dict_cpu['cov_loss_proj']:.4f} std={loss_dict_cpu['std_loss']:.4f}")
                        if args.multistep_k > 1:
                            print(f"  MultiStep: loss={loss_dict_cpu['multistep_jepa_loss']:.4f}")
                        print(f"  EMA tau={ema_updater.tau:.5f}")

        # ── Epoch End ────────────────────────────────────────────────────────────
        # CRITICAL: wait_for_everyone() is a collective barrier — ALL 8 processes
        # must call it simultaneously. It MUST be outside is_main_process blocks
        # or the 7 non-main processes never reach it and main hangs indefinitely.
        if IS_TPU:
            import torch_xla.core.xla_model as xm
            xm.mark_step()                  # flush XLA graph before barrier
            accelerator.wait_for_everyone() # all 8 cores sync here

        if is_main_process:
            # Materialize avg_loss after the barrier (avoids premature device sync)
            avg_loss = (total_loss / len(dataloader)).item() if isinstance(total_loss, torch.Tensor) else total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")

            # Save Checkpoint
            save_path = Path(args.output_dir)
            save_path.mkdir(exist_ok=True)
            if IS_TPU:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), save_path / f"world_model_epoch_{epoch+1}.pt")
            else:
                state_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state_to_save, save_path / f"world_model_epoch_{epoch+1}.pt")

            # Save CSV metrics for this epoch
            if args.logger == "csv" and csv_metrics:
                import csv
                csv_file = save_path / f"metrics_epoch_{epoch+1}.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_metrics[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_metrics)
                csv_metrics = []  # Reset for next epoch

    if is_main_process:
        print("Training Complete!")
        if args.logger == "tensorboard" and logger is not None:
            logger.close()
    if not IS_TPU and world_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    train()
