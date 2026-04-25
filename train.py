import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from pathlib import Path
import argparse
import sys

# Detect project root dynamically
PROJECT_ROOT = Path(__file__).parent.resolve()

# Ensure project root is in path for robust 'src.xxx' imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.world_model import ARCJEPAWorldModel
from src.training.loss import ARCJPELoss
from src.training.ema import EMAUpdater
from src.data.dataset import ARCTrajectoryDataset, create_mock_trajectory
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

def train():
    parser = argparse.ArgumentParser(description="ARC-JEPA Training Loop")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context_ratio", type=float, default=0.7)
    parser.add_argument("--use_curriculum", action="store_true", help="Use curriculum learning for context ratio")
    parser.add_argument("--curriculum_start", type=float, default=0.9, help="Starting context ratio for curriculum")
    parser.add_argument("--curriculum_end", type=float, default=0.5, help="Ending context ratio for curriculum")
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data" / "recordings"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--recon_weight", type=float, default=0.01)
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed for training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    # Research extension flags
    parser.add_argument("--use_vicreg", action="store_true", help="Enable VICReg covariance loss")
    parser.add_argument("--vicreg_weight", type=float, default=1.0, help="Weight for VICReg covariance loss")
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

    # Initialize Distributed environment
    rank, world_size, local_rank = setup_dist(args.local_rank)
    is_main_process = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 1. Setup Data
    data_path = Path(args.data_dir)
    if is_main_process:
        data_path.mkdir(parents=True, exist_ok=True)
    
    # Wait for directory creation if multi-gpu
    if world_size > 1: dist.barrier()
    
    # Check if files exist and generate if necessary (Main process only)
    if is_main_process and not list(data_path.glob("*.jsonl")):
        print(f"No recordings found in {args.data_dir}. Generating real trajectories via ARC-AGI...")
        create_mock_trajectory(args.data_dir, num_trajectories=args.num_trajectories)
        
    # ALL processes wait for generation to finish
    if world_size > 1: dist.barrier()
    
    # CRITICAL: ALL processes re-read the directory after potential generation
    recording_files = list(data_path.glob("*.jsonl"))
    
    if not recording_files:
        recording_files = list(PROJECT_ROOT.glob("**/*.jsonl"))
        
    if not recording_files:
        if is_main_process: print("Error: No .jsonl files found.")
        sys.exit(1)

    dataset = ARCTrajectoryDataset(
        [str(f) for f in recording_files],
        multistep_k=args.multistep_k,
        compute_temporal_masks=args.compute_temporal_masks,
        filter_noops=args.filter_noops
    )

    # Validate ARC compliance
    if is_main_process:
        dataset.validate_arc_compliance()

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    # Use num_workers=0 for distributed to avoid potential multiproc deadlocks on some systems
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=(sampler is None), 
        sampler=sampler,
        num_workers=0 if world_size > 1 else 4,
        pin_memory=True
    )

    # 2. Initialize Model & Components
    model = ARCJEPAWorldModel(
        d_model=512,  # Increased from 256
        num_vit_layers=6,  # Increased from 4
        num_gdn_heads=8,  # Increased from 4
        multistep_k=args.multistep_k
    ).to(device)
    
    # DeepSpeed Integration (Optional)
    if args.deepspeed:
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
    if world_size > 1 and not args.deepspeed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    criterion = ARCJPELoss(
        recon_weight=args.recon_weight,
        use_vicreg=args.use_vicreg,
        vicreg_weight=args.vicreg_weight,
        use_focal=args.use_focal,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        temporal_weight_multiplier=args.temporal_weight
    )
    if not args.deepspeed:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # EMA Updater
    ema_updater = EMAUpdater(
        model.module.online_encoder if hasattr(model, 'module') else model.online_encoder,
        model.module.target_encoder if hasattr(model, 'module') else model.target_encoder,
        tau=0.999
    )

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
        print("Note: First batch may take 2-3 minutes due to Triton kernel compilation.")

    # CSV logging setup
    csv_metrics = []

    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        total_loss = 0

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

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if args.deepspeed:
                outputs = model(batch, context_ratio=context_ratio, use_teacher_forcing=True)
                loss_dict = criterion(outputs, batch)
                loss = loss_dict['loss']
                model.backward(loss)
                model.step()
            else:
                optimizer.zero_grad()
                outputs = model(batch, context_ratio=context_ratio, use_teacher_forcing=True)
                loss_dict = criterion(outputs, batch)
                loss = loss_dict['loss']
                loss.backward()
                optimizer.step()

            ema_updater.update()
            total_loss += loss.item()

            # Compute comprehensive metrics
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    latent_metrics = compute_latent_metrics(outputs['target_latents'])
                    pred_metrics = compute_prediction_metrics(
                        outputs['decoder_logits'],
                        batch['final_state'],
                        batch.get('temporal_mask', None)
                    )
                    grad_metrics = compute_gradient_metrics(model)
                    data_metrics = compute_data_statistics(batch['states'], batch['target_states'])

                    # Aggregate all metrics
                    metrics = {
                        'epoch': epoch + 1,
                        'batch': batch_idx,
                        'lr': optimizer.param_groups[0]['lr'] if not args.deepspeed else args.lr,
                        **loss_dict,
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
                        print(f"  Loss: {loss.item():.4f} | JEPA: {loss_dict['jepa_loss']:.4f} | Recon: {loss_dict['recon_loss']:.4f} | Std: {loss_dict['std_loss']:.4f} | Policy: {loss_dict['policy_loss']:.4f}")
                        print(f"  Latent: std={latent_metrics['latent_std_mean']:.3f} rank={latent_metrics['effective_rank']:.1f} corr={latent_metrics['latent_correlation_max']:.3f}")
                        print(f"  Accuracy: pixel={pred_metrics['pixel_accuracy']:.3f} fg={pred_metrics['foreground_accuracy']:.3f} bg={pred_metrics['background_accuracy']:.3f}")
                        if args.compute_temporal_masks:
                            print(f"  Temporal: changed={pred_metrics['changed_pixel_accuracy']:.3f} unchanged={pred_metrics['unchanged_pixel_accuracy']:.3f}")
                        print(f"  Data: noop={data_metrics['noop_ratio']:.3f} fg_ratio={data_metrics['foreground_ratio']:.3f}")
                        if args.use_vicreg:
                            print(f"  VICReg: cov_loss={loss_dict['cov_loss']:.4f}")
                        if args.multistep_k > 1:
                            print(f"  MultiStep: loss={loss_dict['multistep_jepa_loss']:.4f}")

        if is_main_process:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")

            # Save Checkpoint
            save_path = Path(args.output_dir)
            save_path.mkdir(exist_ok=True)
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
    if world_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    train()
