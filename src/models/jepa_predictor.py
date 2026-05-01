"""
V-JEPA-faithful Transformer Predictor for ARC-JEPA.

CITATION: V-JEPA official predictor.py (VisionTransformerPredictor).
Architecture:
  1. predictor_embed: Linear(d_model → predictor_embed_dim) — projects
     context and creates action-conditioned mask tokens
  2. Learnable mask tokens for target positions
  3. Sinusoidal temporal positional encoding (frozen)
  4. depth × Transformer blocks with self-attention
  5. predictor_norm + predictor_proj back to d_model

Adaptation from V-JEPA:
  - V-JEPA uses spatial mask tokens for hidden patches; we use temporal
    mask tokens for future time steps, conditioned on action embeddings
  - V-JEPA uses 2D sincos spatial positional encoding; we use 1D sincos
    temporal positional encoding
  - Context = GDN-processed temporal features; Targets = future state
    predictions conditioned by actions
"""

import math
import numpy as np

import torch
import torch.nn as nn


def _get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    """Generate 1D sinusoidal positional embeddings.

    Returns: [length, embed_dim] numpy array.
    """
    positions = np.arange(length, dtype=np.float64)
    dim = embed_dim // 2
    omega = np.arange(dim, dtype=np.float64) / dim
    omega = 1.0 / (10000.0 ** omega)
    out = np.outer(positions, omega)  # [length, dim]
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)  # [length, embed_dim]
    if embed_dim % 2 == 1:
        # Pad with a zero column for odd embed_dim
        emb = np.concatenate([emb, np.zeros((length, 1))], axis=1)
    return emb


class JEPAPredictor(nn.Module):
    """
    V-JEPA-faithful Transformer Predictor adapted for ARC World Model.

    During training, context tokens (GDN-processed latents) and
    action-conditioned mask tokens for all future steps are concatenated
    and processed through self-attention blocks. The predictor outputs
    predictions for all target positions in parallel.

    CITATION: V-JEPA predictor.py, lines 23-239.

    Args:
        d_model: Encoder embedding dimension (512).
        predictor_embed_dim: Internal predictor dimension (384).
            CITATION: I-JEPA Table 14 — bottleneck width 384 outperforms
            full width for predictor.
        depth: Number of Transformer blocks (6).
        num_heads: Attention heads (12).
        mlp_ratio: FFN hidden dim = predictor_embed_dim * mlp_ratio.
        max_seq_len: Maximum total sequence length (context + targets).
        init_std: Weight initialization std (0.02, matching V-JEPA).
    """
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,          # kept for API compat, mapped to depth
        num_heads: int = 12,
        bottleneck_dim: int = 384,    # kept for API compat, mapped to predictor_embed_dim
        dropout: float = 0.1,
        max_seq_len: int = 128,       # context + targets
        mlp_ratio: float = 4.0,
        init_std: float = 0.02
    ):
        super().__init__()
        self.d_model = d_model
        predictor_embed_dim = bottleneck_dim  # use bottleneck_dim as predictor dim
        self.predictor_embed_dim = predictor_embed_dim
        depth = num_layers

        # ── Projection: encoder space → predictor space ────────────────────
        # CITATION: V-JEPA predictor.py line 50
        self.predictor_embed = nn.Linear(d_model, predictor_embed_dim, bias=True)

        # ── Action projection to predictor space ───────────────────────────
        self.action_proj = nn.Linear(d_model, predictor_embed_dim, bias=True)

        # ── Learnable mask token ───────────────────────────────────────────
        # CITATION: V-JEPA predictor.py lines 53-59
        # Zero-initialized as in V-JEPA (zero_init_mask_tokens=True default)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # ── Temporal positional encoding (frozen sinusoidal) ───────────────
        # CITATION: V-JEPA predictor.py lines 87-89 (requires_grad=False)
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, predictor_embed_dim),
            requires_grad=False
        )
        # Initialize with sinusoidal encoding
        sincos = _get_1d_sincos_pos_embed(predictor_embed_dim, max_seq_len)
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(sincos).float().unsqueeze(0)
        )

        # ── Transformer blocks ─────────────────────────────────────────────
        # CITATION: V-JEPA predictor.py lines 92-105
        self.predictor_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=predictor_embed_dim,
                nhead=num_heads,
                dim_feedforward=int(predictor_embed_dim * mlp_ratio),
                dropout=dropout,
                batch_first=True,
                activation='gelu',
                norm_first=True  # Pre-norm for training stability
            )
            for _ in range(depth)
        ])

        # ── Output: norm + project back to encoder space ───────────────────
        # CITATION: V-JEPA predictor.py lines 108-109
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, d_model, bias=True)

        # ── Weight initialization ──────────────────────────────────────────
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        """CITATION: V-JEPA predictor.py lines 137-144."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        """Scale down residual branch weights by 1/sqrt(2*layer_id).

        CITATION: V-JEPA predictor.py lines 146-152. Prevents gradient
        explosion at initialization in deep Transformer stacks.
        """
        for layer_id, layer in enumerate(self.predictor_blocks):
            # nn.TransformerEncoderLayer stores self_attn and linear2
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'out_proj'):
                layer.self_attn.out_proj.weight.data.div_(
                    math.sqrt(2.0 * (layer_id + 1))
                )
            if hasattr(layer, 'linear2'):
                layer.linear2.weight.data.div_(
                    math.sqrt(2.0 * (layer_id + 1))
                )

    def forward(
        self,
        context_features: torch.Tensor,
        action_embeds: torch.Tensor,
        step_idx: int = 0
    ) -> torch.Tensor:
        """
        Predict future latent states given GDN-processed context and actions.

        This follows V-JEPA's approach: concatenate context tokens with
        action-conditioned mask tokens, process through self-attention, and
        extract predictions for target positions.

        Args:
            context_features: [B, K, d_model] — GDN-processed context latents
            action_embeds: [B, T_pred, d_model] — action embeddings for each
                prediction step
            step_idx: unused (kept for API compatibility)

        Returns:
            pred_latents: [B, T_pred, d_model] — predicted next-state latents
        """
        B, K, _ = context_features.shape
        T_pred = action_embeds.shape[1]

        # ── 1. Project context to predictor dim ────────────────────────────
        ctx = self.predictor_embed(context_features)  # [B, K, pred_dim]

        # ── 2. Create action-conditioned mask tokens ───────────────────────
        # Each future step gets a learnable mask token + its action embedding
        act = self.action_proj(action_embeds)  # [B, T_pred, pred_dim]
        pred_tokens = self.mask_token.expand(B, T_pred, -1) + act

        # ── 3. Add temporal positional encoding ────────────────────────────
        total_len = K + T_pred
        ctx = ctx + self.predictor_pos_embed[:, :K, :]
        pred_tokens = pred_tokens + self.predictor_pos_embed[:, K:total_len, :]

        # ── 4. Concatenate context + prediction tokens ─────────────────────
        # CITATION: V-JEPA predictor.py line 221
        x = torch.cat([ctx, pred_tokens], dim=1)  # [B, K + T_pred, pred_dim]

        # ── 5. Self-attention over full sequence ───────────────────────────
        # CITATION: V-JEPA predictor.py lines 231-232
        for blk in self.predictor_blocks:
            x = blk(x)

        # ── 6. Normalize and project back ──────────────────────────────────
        # CITATION: V-JEPA predictor.py lines 233, 237
        x = self.predictor_norm(x)

        # Extract only prediction tokens (skip context)
        x = x[:, K:]  # [B, T_pred, pred_dim]

        # Project back to encoder space
        x = self.predictor_proj(x)  # [B, T_pred, d_model]

        return x

    def forward_step(
        self,
        s_t: torch.Tensor,
        action_embed: torch.Tensor,
        gdn_state_summary: torch.Tensor,
        step_idx: int = 0
    ) -> torch.Tensor:
        """
        O(1) MEMORY single-step prediction for MCTS rollouts.

        Instead of self-attention over a growing context (O(N) memory),
        this method uses a fixed-size GDN state summary as context.
        The GDN recurrent state S ∈ ℝ^{HV×K×V} compresses the entire
        history into a fixed-size matrix, which we project into a small
        number of "summary tokens" that serve as the predictor's context.

        Memory per call: O(1) — no sequence buffers, no KV-cache growth.

        Args:
            s_t: [B, d_model] — current state latent (single step)
            action_embed: [B, d_model] — action embedding for this step
            gdn_state_summary: [B, d_model] — GDN output for current step
                (from GDNSequenceModel.step())
            step_idx: temporal position index for positional encoding

        Returns:
            pred_latent: [B, d_model] — predicted next-state latent
        """
        B = s_t.shape[0]

        # Create fixed-size context: [current_state, gdn_summary] = 2 tokens
        ctx_tokens = torch.stack([s_t, gdn_state_summary], dim=1)  # [B, 2, d_model]
        ctx = self.predictor_embed(ctx_tokens)  # [B, 2, pred_dim]

        # Create single action-conditioned mask token
        act = self.action_proj(action_embed.unsqueeze(1))  # [B, 1, pred_dim]
        pred_token = self.mask_token.expand(B, 1, -1) + act  # [B, 1, pred_dim]

        # Add positional encoding (based on step_idx)
        ctx = ctx + self.predictor_pos_embed[:, :2, :]
        pred_token = pred_token + self.predictor_pos_embed[:, 2 + step_idx:3 + step_idx, :]

        # Concatenate: [2 context tokens, 1 prediction token] = 3 tokens total
        x = torch.cat([ctx, pred_token], dim=1)  # [B, 3, pred_dim]

        # Self-attention over 3 tokens — O(1) cost
        for blk in self.predictor_blocks:
            x = blk(x)

        x = self.predictor_norm(x)

        # Extract prediction token (last position)
        x = x[:, -1:]  # [B, 1, pred_dim]
        x = self.predictor_proj(x).squeeze(1)  # [B, d_model]

        return x

    def forward_multistep(
        self,
        s_t: torch.Tensor,
        action_embeds: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """
        Multi-step autoregressive rollout for auxiliary loss (training).

        Uses autoregressive chaining: predict step t+1, feed back as context
        for step t+2, etc. This tests the predictor's ability to chain
        predictions without error accumulation.

        NOTE: This method uses the full ``forward()`` with growing context.
        For O(1) inference (MCTS), use ``forward_step()`` instead.

        Args:
            s_t: Initial state [B, d_model] — single time-step
            action_embeds: [B, k, d_model] — action embeddings for k steps
            k: Number of steps to predict

        Returns:
            predictions: [B, k, d_model] predicted latents
        """
        predictions = []
        # Start with a single context token
        context = s_t.unsqueeze(1)  # [B, 1, d_model]

        for i in range(k):
            z_a = action_embeds[:, i:i+1, :]  # [B, 1, d_model]

            # Predict single next step
            s_next = self.forward(context, z_a, step_idx=i)  # [B, 1, d_model]

            predictions.append(s_next.squeeze(1))

            # Append prediction to context for next step
            context = torch.cat([context, s_next], dim=1)  # [B, i+2, d_model]

        return torch.stack(predictions, dim=1)  # [B, k, d_model]
