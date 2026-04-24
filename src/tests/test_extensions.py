#!/usr/bin/env python3
"""
Quick test script to verify the research extensions are properly integrated.
Tests each component independently before running full training.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.loss_components import VICRegCovarianceLoss, FocalLoss, TemporalSpatialMask
from src.training.metrics import (
    compute_latent_metrics,
    compute_prediction_metrics,
    compute_gradient_metrics,
    compute_data_statistics
)
from src.training.loss import ARCJPELoss
from src.models.jepa_predictor import JEPAPredictor

def test_vicreg_loss():
    print("Testing VICReg Covariance Loss...")
    vicreg = VICRegCovarianceLoss()

    # Test with 3D tensor [B, T, D]
    latents = torch.randn(4, 8, 256)
    loss = vicreg(latents)
    assert loss.item() >= 0, "VICReg loss should be non-negative"
    print(f"  ✓ VICReg loss: {loss.item():.4f}")

def test_focal_loss():
    print("Testing Focal Loss...")
    focal = FocalLoss(alpha=0.25, gamma=2.0)

    # Test with logits [B, C, H, W] and targets [B, H, W]
    logits = torch.randn(2, 16, 64, 64)
    targets = torch.randint(0, 16, (2, 64, 64))

    loss = focal(logits, targets)
    assert loss.item() >= 0, "Focal loss should be non-negative"
    print(f"  ✓ Focal loss: {loss.item():.4f}")

    # Test with spatial mask
    mask = torch.rand(2, 64, 64)
    loss_masked = focal(logits, targets, mask)
    assert loss_masked.item() >= 0, "Masked focal loss should be non-negative"
    print(f"  ✓ Focal loss (masked): {loss_masked.item():.4f}")

def test_temporal_mask():
    print("Testing Temporal Spatial Mask...")
    mask_fn = TemporalSpatialMask(changed_weight=10.0, unchanged_weight=1.0)

    # Create states with some changes
    states = torch.randint(0, 16, (2, 8, 64, 64))
    target_states = states.clone()
    target_states[:, :, 10:20, 10:20] = 5  # Change a region

    mask = mask_fn(states, target_states)
    assert mask.shape == states.shape, "Mask shape should match input"
    assert mask.max() == 10.0, "Changed pixels should have weight 10.0"
    assert mask.min() == 1.0, "Unchanged pixels should have weight 1.0"
    print(f"  ✓ Temporal mask: shape={mask.shape}, max={mask.max():.1f}, min={mask.min():.1f}")

def test_latent_metrics():
    print("Testing Latent Metrics...")
    latents = torch.randn(4, 8, 256)

    metrics = compute_latent_metrics(latents)
    assert 'latent_std_mean' in metrics
    assert 'effective_rank' in metrics
    assert 'latent_correlation_max' in metrics
    print(f"  ✓ Latent metrics: std={metrics['latent_std_mean']:.3f}, rank={metrics['effective_rank']:.1f}")

def test_prediction_metrics():
    print("Testing Prediction Metrics...")
    logits = torch.randn(2, 16, 64, 64)
    targets = torch.randint(0, 16, (2, 64, 64))

    metrics = compute_prediction_metrics(logits, targets)
    assert 'pixel_accuracy' in metrics
    assert 'foreground_accuracy' in metrics
    assert 'background_accuracy' in metrics
    print(f"  ✓ Prediction metrics: acc={metrics['pixel_accuracy']:.3f}")

def test_multistep_predictor():
    print("Testing Multi-Step Predictor...")
    predictor = JEPAPredictor(d_model=256)

    # Test single-step (backward compatibility)
    s_t = torch.randn(2, 4, 256)
    z_a = torch.randn(2, 4, 256)
    output = predictor(s_t, z_a)
    assert output.shape == (2, 4, 256), "Single-step output shape mismatch"
    print(f"  ✓ Single-step prediction: shape={output.shape}")

    # Test multi-step
    s_t_init = torch.randn(2, 256)
    action_embeds = torch.randn(2, 3, 256)
    output_multi = predictor.forward_multistep(s_t_init, action_embeds, k=3)
    assert output_multi.shape == (2, 3, 256), "Multi-step output shape mismatch"
    print(f"  ✓ Multi-step prediction (k=3): shape={output_multi.shape}")

def test_integrated_loss():
    print("Testing Integrated Loss Function...")

    # Test with all extensions enabled
    criterion = ARCJPELoss(
        recon_weight=0.1,
        use_vicreg=True,
        vicreg_weight=1.0,
        use_focal=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
        temporal_weight_multiplier=10.0
    )

    # Mock outputs and targets
    outputs = {
        'pred_latents': torch.randn(2, 8, 256),
        'target_latents': torch.randn(2, 8, 256),
        'decoder_logits': torch.randn(2, 16, 64, 64)
    }

    targets = {
        'final_state': torch.randint(0, 16, (2, 64, 64)),
        'states': torch.randint(0, 16, (2, 8, 64, 64)),
        'target_states': torch.randint(0, 16, (2, 8, 64, 64))
    }

    loss_dict = criterion(outputs, targets)

    assert 'loss' in loss_dict
    assert 'jepa_loss' in loss_dict
    assert 'recon_loss' in loss_dict
    assert 'std_loss' in loss_dict
    assert 'cov_loss' in loss_dict
    assert 'focal_loss' in loss_dict

    print(f"  ✓ Total loss: {loss_dict['loss'].item():.4f}")
    print(f"  ✓ JEPA loss: {loss_dict['jepa_loss']:.4f}")
    print(f"  ✓ Recon loss: {loss_dict['recon_loss']:.4f}")
    print(f"  ✓ Std loss: {loss_dict['std_loss']:.4f}")
    print(f"  ✓ Cov loss: {loss_dict['cov_loss']:.4f}")
    print(f"  ✓ Focal loss: {loss_dict['focal_loss']:.4f}")

if __name__ == "__main__":
    print("=" * 60)
    print("ARC-JEPA Research Extensions - Component Tests")
    print("=" * 60)

    try:
        test_vicreg_loss()
        test_focal_loss()
        test_temporal_mask()
        test_latent_metrics()
        test_prediction_metrics()
        test_multistep_predictor()
        test_integrated_loss()

        print("\n" + "=" * 60)
        print("✓ All tests passed! Extensions are properly integrated.")
        print("=" * 60)
        print("\nYou can now run training with:")
        print("  python train.py --use_vicreg --multistep_k 3 --use_focal --compute_temporal_masks --filter_noops")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
