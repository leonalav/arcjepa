import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
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

def train():
    parser = argparse.ArgumentParser(description="ARC-JEPA Training Loop")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context_ratio", type=float, default=0.3)
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data" / "recordings"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--recon_weight", type=float, default=0.1)
    args = parser.parse_args()

    # 1. Setup Data
    data_path = Path(args.data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    recording_files = list(data_path.glob("*.jsonl"))
    
    # If no data, create some mock trajectories for sanity check
    if not recording_files:
        print(f"No recordings found in {args.data_dir}. Generating real trajectories via ARC-AGI...")
        create_mock_trajectory(args.data_dir, num_trajectories=5)
        
        recording_files = list(data_path.glob("*.jsonl"))
        if not recording_files:
            # Fallback to searching recursively from root if still empty
            recording_files = list(PROJECT_ROOT.glob("**/*.jsonl"))
            
    if not recording_files:
        print("Error: No .jsonl files found even after generation attempt.")
        sys.exit(1)

    dataset = ARCTrajectoryDataset([str(f) for f in recording_files])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 2. Initialize Model & Components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ARCJEPAWorldModel(d_model=256).to(device)
    
    criterion = ARCJPELoss(recon_weight=args.recon_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # EMA Updater for Target Encoder
    ema_updater = EMAUpdater(model.online_encoder, model.target_encoder, tau=0.999)

    # 3. Training Loop
    model.train()
    print(f"Starting training on {device}...")
    print(f"Dataset size: {len(dataset)}")
    
    for epoch in range(args.epochs):
        total_loss = 0
        total_jepa = 0
        total_recon = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward Pass with multi-step rollout
            outputs = model(batch, context_ratio=args.context_ratio)
            
            # Loss Calculation
            loss_dict = criterion(outputs, batch)
            loss = loss_dict['loss']
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            # EMA Update
            ema_updater.update()
            
            total_loss += loss.item()
            total_jepa += loss_dict['jepa_loss'].item()
            total_recon += loss_dict['recon_loss'].item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} (JEPA: {loss_dict['jepa_loss']:.4f}, Recon: {loss_dict['recon_loss']:.4f})")
        
        avg_loss = total_loss / len(dataloader)
        avg_jepa = total_jepa / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f} (JEPA: {avg_jepa:.4f}, Recon: {avg_recon:.4f})")
        
        # Save Checkpoint
        save_path = Path(args.output_dir)
        save_path.mkdir(exist_ok=True)
        torch.save(model.state_dict(), save_path / f"world_model_epoch_{epoch+1}.pt")

    print("Training Complete!")

if __name__ == "__main__":
    train()
