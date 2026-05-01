#!/bin/bash
# ARC-JEPA Training Script — V-JEPA-faithful overhaul
#
# Changes from previous version:
#   - Removed all VICReg arguments (vicreg_weight, var_weight, proj_cov_weight, use_vicreg)
#   - Added reg_coeff (V-JEPA variance regularization coefficient)
#   - Added warmup_epochs and weight_decay
#   - Added --filter_noops to eliminate static data
#   - JEPA loss is now MSE in encoder space (not L1, not in projector space)
#   - EMA momentum anneals per-step (not per-epoch)
#   - Cosine LR schedule with linear warmup

# ─── DeepSpeed (GPU) ─────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deepspeed train.py \
  --deepspeed \
  --batch_size 8 \
  --epochs 10 \
  --lr 1.5e-4 \
  --warmup_epochs 1 \
  --weight_decay 0.05 \
  --context_ratio 0.7 \
  --recon_weight 0.01 \
  --reg_coeff 1.0 \
  --multistep_k 3 \
  --use_focal \
  --compute_temporal_masks \
  --filter_noops \
  --logger csv \
  --num_trajectories 1000

# ─── TPU v5e-8 ───────────────────────────────────────────────────────────────
export TF_CPP_MIN_LOG_LEVEL=3
export PT_XLA_DEBUG=0
export XLA_PERSISTENT_CACHE_DIR=/tmp/xla_cache
mkdir -p /tmp/xla_cache
unset TPU_PROCESS_ADDRESSES
unset CLOUD_TPU_TASK_ID
export PJRT_DEVICE=TPU

accelerate launch --num_processes 8 --mixed_precision bf16 train.py \
  --hf_repo_id "leonidas123/arc3data" \
  --batch_size 4 \
  --epochs 100 \
  --lr 1.5e-4 \
  --warmup_epochs 5 \
  --weight_decay 0.05 \
  --context_ratio 0.7 \
  --recon_weight 0.01 \
  --reg_coeff 1.0 \
  --multistep_k 3 \
  --filter_noops \
  --logger csv
