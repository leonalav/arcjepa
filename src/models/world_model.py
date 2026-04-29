"""
# 1. Strip the rogue multi-host environment variables
unset TPU_PROCESS_ADDRESSES
unset CLOUD_TPU_TASK_ID

# 2. Set the modern XLA backend
export PJRT_DEVICE=TPU

# 3. Launch
accelerate launch --num_processes 8 train.py \
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
  --logger csv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional, Dict, Any

from .embeddings import GridEmbedding, ActionEmbedding, PositionalEncoding2D
from .spatial_encoder import DiscreteViT
from .sequence_model import GDNSequenceModel
from .jepa_predictor import JEPAPredictor
from .decoder import GridDecoder

class ARCJEPAWorldModel(nn.Module):
    """
    ARC-JEPA World Model: Combines spatial encoding, temporal sequence modeling,
    and latent prediction with EMA target encoders.
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        num_vit_layers: int = 4,
        num_gdn_heads: int = 4,
        tau: float = 0.999,
        multistep_k: int = 1,
        predictor_layers: int = 6,
        predictor_bottleneck: int = 384
    ):
        super().__init__()
        self.d_model = d_model
        self.multistep_k = multistep_k

        # Shared Embeddings
        self.grid_embed = GridEmbedding(d_model)
        self.pos_embed = PositionalEncoding2D(d_model)
        self.action_embed = ActionEmbedding(d_model)

        # Encoders
        self.online_encoder = DiscreteViT(d_model, nhead=n_heads, num_layers=num_vit_layers)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Temporal / Sequence Model
        self.gdn = GDNSequenceModel(d_model, n_heads=num_gdn_heads)

        # Predictor (Lightweight MLP with bottleneck)
        self.predictor = JEPAPredictor(
            d_model=d_model,
            num_layers=predictor_layers,
            bottleneck_dim=predictor_bottleneck
        )

        # Final State Decoder
        self.decoder = GridDecoder(d_model)

        # Policy Head: s_t -> (action_logits, x_logits, y_logits)
        # Enables AlphaZero-style pruning during latent search
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 9 + 64 + 64) # 9 actions + 64x + 64y
        )

        # VICReg Projector (for disentangling features in a higher-dimensional space)
        # This prevents the core latents from being too constrained by the covariance penalty
        self.projector = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )

    def encode(self, grids: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """
        grids: [Batch, T, 64, 64]
        encoder: online_encoder or target_encoder
        Returns: [Batch, T, d_model] latent states

        With 4×4 patches, each frame has 256 patch tokens + CLS = 257 tokens.
        Attention per frame = 257² × 2 bytes (bf16) × 8 heads ≈ 1 MB.
        Even at B×T = 4×64 = 256 frames: 256 MB total attention — trivially fits
        in 16 GB HBM.  No micro-batching loop required, and XLA compiles a single
        clean graph with no unrolled iterations.
        """
        b, t, h, w = grids.shape
        # Flatten time: [B*T, H, W]
        grids_flat = grids.reshape(b * t, h, w)

        # Patch embedding: [B*T, Ph, Pw, d_model]  (Ph=Pw=16 for 4×4 patches)
        x = self.grid_embed(grids_flat)

        # 2D positional encoding at patch-grid resolution: [Ph, Pw, d_model]
        Ph, Pw = x.shape[1], x.shape[2]
        pos = self.pos_embed(Ph, Pw)
        x = x + pos.unsqueeze(0)  # broadcast over batch

        # Single-pass through ViT: [B*T, d_model]
        latents = encoder(x)

        return latents.reshape(b, t, self.d_model)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        context_ratio: float = 0.7,
        use_teacher_forcing: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with teacher forcing and no GDN feedback loop.

        Key changes from original:
        - Teacher forcing: use ground truth states as predictor input
        - GDN only for context processing, NOT in prediction loop
        - Store RAW predictor outputs for loss (no GDN processing)
        - Increased default context_ratio from 0.3 to 0.7

        Args:
            batch: Dictionary with states, actions, coords, target_states
            context_ratio: Fraction of sequence used as context (default 0.7)
            use_teacher_forcing: Use ground truth states during training (default True)

        Returns:
            Dictionary with pred_latents, target_latents, decoder_logits, policy_logits
        """
        states = batch['states']              # [B, T, 64, 64]
        actions = batch['actions']            # [B, T]
        cx, cy = batch['coords_x'], batch['coords_y']
        target_grids = batch['target_states'] # [B, T, 64, 64]

        B, T = states.shape[:2]
        K = max(1, int(T * context_ratio))

        # ═══════════════════════════════════════════════════════
        # PHASE 1: CONTEXT ENCODING (with GDN)
        # ═══════════════════════════════════════════════════════

        # Encode context states with online encoder
        s_context = self.encode(states[:, :K], self.online_encoder)  # [B, K, d_model]

        # Process context with GDN for temporal features
        s_context_features, _ = self.gdn(s_context, use_cache=True)

        # ═══════════════════════════════════════════════════════
        # PHASE 2: TEACHER-FORCED PREDICTION (NO GDN)
        # ═══════════════════════════════════════════════════════

        if use_teacher_forcing:
            # TEACHER FORCING: Use ground truth states as input
            all_states_encoded = self.encode(states[:, :T-1], self.online_encoder)  # [B, T-1, d_model]

            # Vectorized action embeddings: embed the full prediction window in one op.
            # Slicing before embedding avoids a Python for-loop that would add T separate
            # nodes to the XLA graph.
            action_embeds = self.action_embed(
                actions[:, K-1:T-1],   # [B, T-K]
                cx[:, K-1:T-1],
                cy[:, K-1:T-1]
            )  # [B, T-K, d_model]

            # Batch prediction
            s_input = all_states_encoded[:, K-1:T-1]         # [B, T-K, d_model]
            pred_latents = self.predictor(s_input, action_embeds, step_idx=K-1)  # [B, T-K, d_model]

            # Multi-step prediction (auxiliary loss)
            if self.multistep_k > 1 and K-1 + self.multistep_k <= T:
                s_init = all_states_encoded[:, K-1]           # [B, d_model]
                action_embeds_k = self.action_embed(
                    actions[:, K-1 : K-1+self.multistep_k],
                    cx[:, K-1 : K-1+self.multistep_k],
                    cy[:, K-1 : K-1+self.multistep_k]
                )  # [B, multistep_k, d_model]

                multistep_pred_latents = self.predictor.forward_multistep(
                    s_init, action_embeds_k, self.multistep_k
                )
            else:
                multistep_pred_latents = None

        else:
            # AUTOREGRESSIVE: Use previous predictions (inference mode)
            pred_latents = []
            curr_state = s_context_features[:, -1]  # [B, d_model]

            for t in range(K-1, T-1):
                z_a = self.action_embed(actions[:, t], cx[:, t], cy[:, t])

                # Predict next state (NO GDN processing)
                s_next_pred = self.predictor(
                    curr_state.unsqueeze(1),
                    z_a.unsqueeze(1),
                    step_idx=t
                ).squeeze(1)

                pred_latents.append(s_next_pred)
                curr_state = s_next_pred  # Autoregressive

            pred_latents = torch.stack(pred_latents, dim=1)
            multistep_pred_latents = None

        # ═══════════════════════════════════════════════════════
        # PHASE 3: TARGET ENCODING (EMA)
        # ═══════════════════════════════════════════════════════

        with torch.no_grad():
            # Encode target states with target encoder (raw embeddings)
            # Slice K-1:T-1 (grids[K...T-1]) to match pred_latents
            s_next_target = self.encode(target_grids[:, K-1:T-1], self.target_encoder)

            # Normalise target representations over the feature dimension.
            # CITATION: V-JEPA official vjepa/train.py, forward_target(), line 426:
            #   h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            # This stabilises the target representation scale as EMA tau
            # anneals toward 1.0 (target encoder nearly frozen) and prevents
            # the JEPA MSE loss scale from drifting across training.
            s_next_target = F.layer_norm(s_next_target, (s_next_target.size(-1),))

            # Multi-step targets
            if self.multistep_k > 1 and K - 1 + self.multistep_k <= T:
                multistep_target_latents = self.encode(
                    target_grids[:, K-1 : K-1+self.multistep_k],
                    self.target_encoder
                )
                multistep_target_latents = F.layer_norm(
                    multistep_target_latents,
                    (multistep_target_latents.size(-1),)
                )
            else:
                multistep_target_latents = None

        # ═══════════════════════════════════════════════════════
        # PHASE 4: AUXILIARY OUTPUTS
        # ═══════════════════════════════════════════════════════

        # Decode final predicted state
        final_state_logits = self.decoder(pred_latents[:, -1])

        # Policy logits
        policy_logits = self.policy_head(pred_latents)

        # VICReg Projection on PRED latents only (online encoder → projector).
        # projected_pred receives gradients from the covariance loss on
        # projector outputs, keeping the projector's 1024D space decorrelated.
        projected_pred_latents = self.projector(pred_latents)

        # CRITICAL: Projector target path must have NO gradients.
        # Previously self.projector(s_next_target) was called outside no_grad,
        # giving the projector gradient signal from the EMA-frozen target path.
        # This trained the projector to map ANY input close to the prediction —
        # a direct collapse shortcut. Wrapping in no_grad eliminates it.
        # CITATION: Official I-JEPA/V-JEPA has no projector at all; the loss
        # is computed directly in encoder space. This is the closest safe
        # approximation when keeping the projector for the cov_proj term.
        with torch.no_grad():
            projected_target_latents = self.projector(s_next_target)

        output = {
            'pred_latents': pred_latents,
            'target_latents': s_next_target,
            'projected_pred_latents': projected_pred_latents,
            'projected_target_latents': projected_target_latents,
            'decoder_logits': final_state_logits,
            'policy_logits': policy_logits
        }

        if multistep_pred_latents is not None:
            output['multistep_pred_latents'] = multistep_pred_latents
            output['multistep_target_latents'] = multistep_target_latents

        return output
