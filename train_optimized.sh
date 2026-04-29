#!/bin/bash
# Memory-optimized training script for ARC-JEPA
#
# Collapse fix (2026-04-29):
#   - var_weight=25.0   : VICReg variance on raw latents (was implicit 1.0)
#   - vicreg_weight=25.0: VICReg covariance on raw latents (was 25 but misdirected)
#   - proj_cov_weight=5 : Covariance on projector outputs (keeps projector from collapsing)
#   - JEPA MSE now computed in projector space (loss.py); not a flag, auto-active
#   - EMA tau anneals 0.996→1.0 per I-JEPA paper (ema.py); not a flag, auto-active

# ─── DeepSpeed (GPU) ─────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deepspeed train.py \
  --deepspeed \
  --batch_size 8 \
  --epochs 10 \
  --lr 1e-4 \
  --context_ratio 0.7 \
  --recon_weight 0.01 \
  --use_vicreg \
  --vicreg_weight 25.0 \
  --var_weight 25.0 \
  --proj_cov_weight 5.0 \
  --multistep_k 3 \
  --use_focal \
  --compute_temporal_masks \
  --filter_noops \
  --logger csv \
  --num_trajectories 1000

# ─── TPU v5e-8 ───────────────────────────────────────────────────────────────
export TF_CPP_MIN_LOG_LEVEL=3        # 0=DEBUG 1=INFO 2=WARNING 3=ERROR (suppress XLA C++ noise)
export PT_XLA_DEBUG=0                 # 0 disables Compilation/Execution Analysis + pt-xla-profiler warnings
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
  --recon_weight 0.01 \
  --use_vicreg \
  --vicreg_weight 25.0 \
  --var_weight 25.0 \
  --proj_cov_weight 5.0 \
  --multistep_k 3 \
  --use_focal \
  --compute_temporal_masks \
  --filter_noops \
  --logger csv
