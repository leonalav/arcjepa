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


# For TPUs
export TF_CPP_MIN_LOG_LEVEL=0
export PT_XLA_DEBUG=1
export XLA_PERSISTENT_CACHE_DIR=/tmp/xla_cache
mkdir -p /tmp/xla_cache
unset TPU_PROCESS_ADDRESSES
unset CLOUD_TPU_TASK_ID
export PJRT_DEVICE=TPU
accelerate launch --num_processes 8 --mixed_precision bf16 train.py \
  --hf_repo_id "leonidas123/arc3data" \
  --batch_size 4 \
  --epochs 10 \
  --lr 1e-4 \
  --context_ratio 0.7 \
  --recon_weight 1.0 \
  --use_vicreg \
  --vicreg_weight 25.0 \
  --multistep_k 3 \
  --use_focal \
  --compute_temporal_masks \
  --logger csv
