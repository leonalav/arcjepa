"""
ARC-JEPA World Model — V-JEPA-faithful architecture.

Combines:
  - Spatial encoder (DiscreteViT) for per-frame encoding
  - Temporal model (GDN in Transformer block) for context processing
  - Transformer predictor (V-JEPA style) for next-state prediction
  - EMA target encoder for self-supervised targets
  - Grid decoder for pixel-level reconstruction
  - Policy head for AlphaZero-style action prediction

Key design decisions:
  1. GDN processes context latents and feeds into predictor (C1 fix)
  2. Single encoding pass — no redundant re-encoding (D2 fix)
  3. No projector — JEPA loss in native encoder space (A1 fix)
  4. Target latents are layer-normalized (V-JEPA faithful)
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

        # Temporal / Sequence Model (GDN wrapped in Transformer block)
        self.gdn = GDNSequenceModel(d_model, n_heads=num_gdn_heads)

        # Predictor — V-JEPA Transformer Predictor
        # Context features from GDN attend to action-conditioned mask tokens
        self.predictor = JEPAPredictor(
            d_model=d_model,
            num_layers=predictor_layers,
            bottleneck_dim=predictor_bottleneck,
            num_heads=max(1, predictor_bottleneck // 32),  # ~12 heads for 384
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
        Forward pass with teacher forcing.

        Architecture flow:
          1. Encode all input states with online encoder (single pass)
          2. Process context window through GDN for temporal features
          3. Predict future states using Transformer predictor with
             GDN context + action-conditioned mask tokens
          4. Encode target states with EMA target encoder

        Args:
            batch: Dictionary with states, actions, coords, target_states
            context_ratio: Fraction of sequence used as context (default 0.7)
            use_teacher_forcing: Use ground truth states during training

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
        # PHASE 1: ENCODE ALL INPUT STATES (single pass — D2 fix)
        # ═══════════════════════════════════════════════════════

        # Encode all T-1 input states with online encoder in one shot.
        # Previously encoded context separately then re-encoded for
        # prediction — wasting compute.
        all_online_latents = self.encode(states[:, :T-1], self.online_encoder)  # [B, T-1, d_model]

        # ═══════════════════════════════════════════════════════
        # PHASE 2: GDN CONTEXT PROCESSING (C1 fix — GDN awakened)
        # ═══════════════════════════════════════════════════════

        # Process context window through GDN for temporal features.
        # The GDN's Transformer block (pre-norm + residual + FFN) produces
        # context representations with temporal coherence.
        context_latents = all_online_latents[:, :K]          # [B, K, d_model]
        context_features, _ = self.gdn(context_latents, use_cache=True)  # [B, K, d_model]

        # ═══════════════════════════════════════════════════════
        # PHASE 3: TRANSFORMER PREDICTION
        # ═══════════════════════════════════════════════════════

        if use_teacher_forcing:
            # Action embeddings for the prediction window
            action_embeds = self.action_embed(
                actions[:, K-1:T-1],   # [B, T-K]
                cx[:, K-1:T-1],
                cy[:, K-1:T-1]
            )  # [B, T-K, d_model]

            # Transformer predictor: GDN context + action-conditioned targets
            # Context features attend to mask tokens via self-attention,
            # enabling each prediction to leverage the full temporal context.
            pred_latents = self.predictor(
                context_features,   # [B, K, d_model] — GDN-processed context
                action_embeds,      # [B, T-K, d_model] — action conditioning
            )  # [B, T-K, d_model]

            # Multi-step prediction (auxiliary loss)
            if self.multistep_k > 1 and K-1 + self.multistep_k <= T:
                s_init = all_online_latents[:, K-1]  # [B, d_model]
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
            # Start with GDN-processed context as initial context
            context_for_ar = context_features  # [B, K, d_model]

            for t in range(K-1, T-1):
                z_a = self.action_embed(
                    actions[:, t], cx[:, t], cy[:, t]
                ).unsqueeze(1)  # [B, 1, d_model]

                # Predict next state using full context
                s_next_pred = self.predictor(
                    context_for_ar,  # [B, ctx_len, d_model]
                    z_a,             # [B, 1, d_model]
                ).squeeze(1)  # [B, d_model]

                pred_latents.append(s_next_pred)

                # Expand context with prediction for next step
                context_for_ar = torch.cat(
                    [context_for_ar, s_next_pred.unsqueeze(1)], dim=1
                )

            pred_latents = torch.stack(pred_latents, dim=1)
            multistep_pred_latents = None

        # ═══════════════════════════════════════════════════════
        # PHASE 4: TARGET ENCODING (EMA)
        # ═══════════════════════════════════════════════════════

        with torch.no_grad():
            # Encode target states with target encoder
            s_next_target = self.encode(target_grids[:, K-1:T-1], self.target_encoder)

            # Layer-normalize target representations
            # CITATION: V-JEPA official vjepa/train.py, forward_target(), line 426:
            #   h = F.layer_norm(h, (h.size(-1),))
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
        # PHASE 5: AUXILIARY OUTPUTS
        # ═══════════════════════════════════════════════════════

        # Decode final predicted state
        final_state_logits = self.decoder(pred_latents[:, -1])

        # Policy logits
        policy_logits = self.policy_head(pred_latents)

        output = {
            'pred_latents': pred_latents,
            'target_latents': s_next_target,
            'decoder_logits': final_state_logits,
            'policy_logits': policy_logits
        }

        if multistep_pred_latents is not None:
            output['multistep_pred_latents'] = multistep_pred_latents
            output['multistep_target_latents'] = multistep_target_latents

        return output
