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
    parser.add_argument("--context_ratio", type=float, default=0.3)
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data" / "recordings"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--recon_weight", type=float, default=0.1)
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed for training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
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
    
    recording_files = list(data_path.glob("*.jsonl"))
    
    if not recording_files and is_main_process:
        print(f"No recordings found in {args.data_dir}. Generating real trajectories via ARC-AGI...")
        create_mock_trajectory(args.data_dir, num_trajectories=5)
        recording_files = list(data_path.glob("*.jsonl"))
        
    if world_size > 1: dist.barrier()
    if not recording_files:
        recording_files = list(PROJECT_ROOT.glob("**/*.jsonl"))
        
    if not recording_files:
        if is_main_process: print("Error: No .jsonl files found.")
        sys.exit(1)

    dataset = ARCTrajectoryDataset([str(f) for f in recording_files])
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
    model = ARCJEPAWorldModel(d_model=256).to(device)
    
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
    
    criterion = ARCJPELoss(recon_weight=args.recon_weight)
    if not args.deepspeed:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # EMA Updater
    ema_updater = EMAUpdater(
        model.module.online_encoder if hasattr(model, 'module') else model.online_encoder, 
        model.module.target_encoder if hasattr(model, 'module') else model.target_encoder, 
        tau=0.999
    )

    # 3. Training Loop
    model.train()
    if is_main_process:
        print(f"Starting training on {world_size} GPU(s) using {'DeepSpeed' if args.deepspeed else 'DDP'}...")
        print(f"Dataset size: {len(dataset)}")
        print("Note: First batch may take 2-3 minutes due to Triton kernel compilation.")
    
    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if is_main_process and batch_idx == 0:
                print(f"Processing first batch of epoch {epoch+1}...")
            
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            if args.deepspeed:
                outputs = model(batch, context_ratio=args.context_ratio)
                loss_dict = criterion(outputs, batch)
                loss = loss_dict['loss']
                model.backward(loss)
                model.step()
            else:
                optimizer.zero_grad()
                outputs = model(batch, context_ratio=args.context_ratio)
                loss_dict = criterion(outputs, batch)
                loss = loss_dict['loss']
                loss.backward()
                optimizer.step()
            
            ema_updater.update()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0 and is_main_process:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
        
        if is_main_process:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")
            
            # Save Checkpoint
            save_path = Path(args.output_dir)
            save_path.mkdir(exist_ok=True)
            state_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_to_save, save_path / f"world_model_epoch_{epoch+1}.pt")

    if is_main_process: print("Training Complete!")
    if world_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    train()
