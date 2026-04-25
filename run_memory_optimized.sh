#!/bin/bash
# Quick fix for OOM error - run this immediately

cd /root/arcjepa

# Enable PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run with minimal memory configuration
python train.py \
    --deepspeed \
    --batch_size 1 \
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

echo "Training started with memory-optimized settings"
echo "Monitor GPU usage with: watch -n 1 nvidia-smi"
