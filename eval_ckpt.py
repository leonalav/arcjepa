#python eval_ckpt.py --checkpoint /root/arcjepa/checkpoints/world_model_epoch_10.pt
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import multiprocessing

# 1. Optimize CPU threads for metric math and background data loading
# Explicitly tell PyTorch to use 80 cores for model matrix math, 
# leaving 16 cores free for the OS and data loaders.
torch.set_num_threads(80)

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.world_model import ARCJEPAWorldModel
from src.training.metrics import compute_prediction_metrics, compute_latent_metrics
from src.data.dataset import FastHFARCDataset
from torch.utils.data import DataLoader

def evaluate(checkpoint_path, hf_repo_id, device):
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    # 2. Initialize Model (MUST match training parameters!)
    model = ARCJEPAWorldModel(
        d_model=512,        # Increased from 256
        num_vit_layers=6,   # Increased from 4
        num_gdn_heads=8,    # Increased from 4
        multistep_k=3       # Matched from your bash script
    )
    
    # Load weights
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Cannot find checkpoint at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Handle DDP/DeepSpeed state dict wrapping if present (removes 'module.' prefix)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # 3. Load Validation Data
    print(f"Loading validation data from {hf_repo_id}...")
    try:
        # Try to load a dedicated validation split if it exists
        dataset = FastHFARCDataset(hf_repo_id=hf_repo_id, split="validation")
    except ValueError:
        print("⚠ 'validation' split not found on HF. Falling back to 'train' split for sanity check.")
        dataset = FastHFARCDataset(hf_repo_id=hf_repo_id, split="train")

    # High-throughput data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=12, 
        pin_memory=False
    )
    
    val_metrics = {"jepa_loss": 0.0, "recon_loss": 0.0, "pixel_acc": 0.0, "rank": 0.0, "corr": 0.0}
    
    print("Starting evaluation loop...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Send data to TPU/Device
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch_device, context_ratio=0.7, use_teacher_forcing=True)
            
            # Tell XLA to execute the graph immediately
            if device.type == 'xla':
                import torch_xla.core.xla_model as xm
                xm.mark_step()

            # Bring results BACK to CPU for metric math. 
            # CRITICAL: If we run SVD (for rank) or Histograms on the TPU, XLA will 
            # graph-break and hang. CPU is perfect for metric reduction.
            pred_lat_cpu = outputs['pred_latents'].cpu()
            tgt_lat_cpu = outputs['target_latents'].cpu()
            logits_cpu = outputs['decoder_logits'].cpu()
            
            # Final state is already on CPU from the dataloader dict
            final_state_cpu = batch['final_state'] 
            
            # Loss computations (on CPU)
            mse = F.mse_loss(pred_lat_cpu, tgt_lat_cpu)
            val_metrics["jepa_loss"] += mse.item()
            
            recon_loss = F.cross_entropy(logits_cpu, final_state_cpu)
            val_metrics["recon_loss"] += recon_loss.item()

            # Metric computations (on CPU)
            pred_metrics = compute_prediction_metrics(logits_cpu, final_state_cpu)
            val_metrics["pixel_acc"] += pred_metrics['pixel_accuracy']
            
            lat_metrics = compute_latent_metrics(tgt_lat_cpu)
            val_metrics["rank"] += lat_metrics['effective_rank']
            val_metrics["corr"] += lat_metrics['latent_correlation_max']
            
            if batch_idx % 5 == 0:
                print(f"  Processed {batch_idx}/{len(dataloader)} batches...")

    # Average metrics
    num_batches = len(dataloader)
    for k in val_metrics:
        val_metrics[k] /= num_batches
        
    # Print Report
    print("\n" + "="*50)
    print(f"VALIDATION RESULTS: {Path(checkpoint_path).name}")
    print("="*50)
    print(f"  JEPA MSE Loss: {val_metrics['jepa_loss']:.4f}")
    print(f"  Recon Loss:    {val_metrics['recon_loss']:.4f}")
    print(f"  Pixel Acc:     {val_metrics['pixel_acc']:.4f}  (1.0 is perfect)")
    print(f"  Latent Rank:   {val_metrics['rank']:.1f}   (Out of 512)")
    print(f"  Latent Corr:   {val_metrics['corr']:.4f}  (Lower is better)")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Exact path to the .pt file")
    parser.add_argument("--hf_repo_id", type=str, default="leonidas123/arc3data")
    args = parser.parse_args()
    
    # Try to grab a TPU core, fallback to CUDA/CPU
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"🚀 Executing on TPU: {device}")
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Executing on: {device}")
    
    evaluate(args.checkpoint, args.hf_repo_id, device)