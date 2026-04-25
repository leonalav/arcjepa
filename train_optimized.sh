#!/bin/bash
# Memory-optimized training script for ARC-JEPA

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use gradient checkpointing and mixed precision
python train.py \
    --deepspeed \
    --batch_size 2 \
    --epochs 100 \
    --lr 1e-4 \
    --context_ratio 0.7 \
    --use_curriculum \
    --recon_weight 0.01 \
    --use_vicreg \
    --multistep_k 1 \
    --filter_noops \
    --num_trajectories 1000 \
    --logger csv
