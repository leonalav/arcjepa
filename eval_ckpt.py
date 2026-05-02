# python eval_ckpt.py --checkpoint /root/arcjepa/checkpoints/world_model_epoch_10.pt
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arc_schema import DEFAULT_NUM_ACTIONS
from src.data.dataset import FastHFARCDataset
from src.models.world_model import ARCJEPAWorldModel
from src.training.metrics import (
    compute_action_metrics,
    compute_efficiency_metrics,
    compute_latent_metrics,
    compute_prediction_metrics,
    compute_terminal_metrics,
)


def _checkpoint_config(checkpoint):
    config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    return {
        'd_model': int(config.get('d_model', 512)),
        'num_vit_layers': int(config.get('num_vit_layers', 6)),
        'num_gdn_heads': int(config.get('num_gdn_heads', 8)),
        'multistep_k': int(config.get('multistep_k', 1)),
        'num_actions': int(config.get('num_actions', DEFAULT_NUM_ACTIONS)),
        'max_games': int(config.get('max_games', 4096)),
        'max_game_families': int(config.get('max_game_families', 512)),
    }


def evaluate(checkpoint_path, hf_repo_id, device):
    print(f"\nLoading checkpoint: {checkpoint_path}")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Cannot find checkpoint at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = _checkpoint_config(checkpoint)
    model = ARCJEPAWorldModel(**cfg)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected checkpoint keys: {len(unexpected)}")
    model.eval().to(device)

    print(f"Loading validation data from {hf_repo_id}...")
    try:
        dataset = FastHFARCDataset(hf_repo_id=hf_repo_id, split="validation", num_actions=cfg['num_actions'])
    except ValueError:
        print("validation split not found on HF; falling back to train split for sanity check.")
        dataset = FastHFARCDataset(hf_repo_id=hf_repo_id, split="train", num_actions=cfg['num_actions'])

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=12, pin_memory=False)
    val_metrics = {
        "jepa_loss": 0.0,
        "recon_loss": 0.0,
        "pixel_acc": 0.0,
        "rank": 0.0,
        "corr": 0.0,
        "action_acc": 0.0,
        "invalid_action_rate": 0.0,
        "terminal_acc": 0.0,
        "efficiency_mae": 0.0,
    }

    print("Starting evaluation loop...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch_device, context_ratio=0.7, use_teacher_forcing=True)
            if device.type == 'xla':
                import torch_xla.core.xla_model as xm
                xm.mark_step()

            pred_lat_cpu = outputs['pred_latents'].cpu()
            tgt_lat_cpu = outputs['target_latents'].cpu()
            logits_cpu = outputs['decoder_logits'].cpu()
            final_state_cpu = batch['final_state']
            seq_mask_cpu = batch.get('seq_mask', None)

            val_metrics["jepa_loss"] += F.mse_loss(pred_lat_cpu, tgt_lat_cpu).item()
            val_metrics["recon_loss"] += F.cross_entropy(logits_cpu, final_state_cpu).item()
            pred_metrics = compute_prediction_metrics(logits_cpu, final_state_cpu, states=batch.get('states', None))
            val_metrics["pixel_acc"] += pred_metrics['pixel_accuracy']
            lat_metrics = compute_latent_metrics(tgt_lat_cpu, seq_mask=seq_mask_cpu)
            val_metrics["rank"] += lat_metrics['effective_rank']
            val_metrics["corr"] += lat_metrics['latent_correlation_max']

            action_metrics = compute_action_metrics(outputs['raw_action_logits'].cpu(), batch['actions'], batch.get('available_actions_mask'), seq_mask=seq_mask_cpu)
            terminal_metrics = compute_terminal_metrics(outputs['terminal_logits'].cpu(), batch.get('terminal', torch.zeros_like(batch['actions'])), seq_mask=seq_mask_cpu)
            efficiency_metrics = compute_efficiency_metrics(outputs['efficiency_pred'].cpu(), batch.get('efficiency_target', torch.zeros_like(batch['actions'], dtype=torch.float32)), seq_mask=seq_mask_cpu)
            val_metrics["action_acc"] += action_metrics['action_accuracy']
            val_metrics["invalid_action_rate"] += action_metrics['invalid_action_rate']
            val_metrics["terminal_acc"] += terminal_metrics['terminal_accuracy']
            val_metrics["efficiency_mae"] += efficiency_metrics['efficiency_mae']

            if batch_idx % 5 == 0:
                print(f"  Processed {batch_idx}/{len(dataloader)} batches...")

    num_batches = max(1, len(dataloader))
    for k in val_metrics:
        val_metrics[k] /= num_batches

    print("\n" + "="*50)
    print(f"VALIDATION RESULTS: {Path(checkpoint_path).name}")
    print("="*50)
    print(f"  JEPA MSE Loss:       {val_metrics['jepa_loss']:.4f}")
    print(f"  Recon Loss:          {val_metrics['recon_loss']:.4f}")
    print(f"  Pixel Acc:           {val_metrics['pixel_acc']:.4f}")
    print(f"  Action Acc:          {val_metrics['action_acc']:.4f}")
    print(f"  Invalid Action Rate: {val_metrics['invalid_action_rate']:.4f}")
    print(f"  Terminal Acc:        {val_metrics['terminal_acc']:.4f}")
    print(f"  Efficiency MAE:      {val_metrics['efficiency_mae']:.4f}")
    print(f"  Latent Rank:         {val_metrics['rank']:.1f}")
    print(f"  Latent Corr:         {val_metrics['corr']:.4f}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hf_repo_id", type=str, default="leonidas123/arc3data")
    args = parser.parse_args()
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"Executing on TPU: {device}")
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Executing on: {device}")
    evaluate(args.checkpoint, args.hf_repo_id, device)
