#!/bin/bash
# Memory-optimized training script for ARC-JEPA

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use gradient checkpointing and mixed precision
deepspeed train.py \
  --deepspeed \
  --batch_size 8 \
  --epochs 10 \
  --lr 1e-4 \
  --context_ratio 0.7 \
  --recon_weight 1.0 \
  --use_vicreg \
  --vicreg_weight 25.0 \
  --multistep_k 3 \
  --use_focal \
  --compute_temporal_masks \
  --filter_noops \
  --logger csv \
  --num_trajectories 1000

