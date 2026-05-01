import torch
import torch.nn as nn
from typing import Optional, Tuple

# --- Backend selection -------------------------------------------------------
# Priority:
#   1. TPU / CPU → gdntpu (pure PyTorch, XLA-safe, no Triton)
#   2. CUDA + FLA installed → FLA Triton kernels (fast GPU training)
#   3. Fallback → gdntpu (always works)
# -----------------------------------------------------------------------------

# gdntpu is always available (pure PyTorch).
from src.models.gdntpu import GatedDeltaNet as TPUGatedDeltaNet

# Optionally import FLA for GPU fast-path.
try:
    from fla.layers import GatedDeltaNet as FLAGatedDeltaNet
    HAS_FLA = True
except ImportError:
    HAS_FLA = False


class GDNSequenceModel(nn.Module):
    """Sequence model using Gated Delta Network wrapped in a proper
    Transformer block.

    The raw GDN layer is an attention-like mechanism. Following standard
    practice (and matching the FLA library's intended usage), we wrap it
    in a full Transformer block:
        x = x + GDN(PreNorm(x))         # attention + residual
        x = x + FFN(PreNorm(x))         # feed-forward + residual

    This provides:
    - Residual connections for stable gradient flow
    - Pre-normalization for training stability
    - Feed-forward network for per-token feature transformation
    - GDN remains O(N) linear complexity (satisfies user constraint)

    On TPU/CPU: uses the pure-PyTorch ``gdntpu.GatedDeltaNet`` (XLA-safe).
    On CUDA (with fla installed): uses ``fla.layers.GatedDeltaNet`` (Triton).

    State-dict keys are identical between both backends, so checkpoints saved
    on TPU can be loaded directly into the CUDA variant and vice-versa.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        is_cuda = torch.cuda.is_available()

        if is_cuda and HAS_FLA:
            # Fast GPU path via Triton kernels.
            self.gdn = FLAGatedDeltaNet(
                hidden_size=d_model,
                num_heads=n_heads,
                **kwargs,
            )
            self._backend = "fla"
        else:
            # TPU / CPU path — pure PyTorch, fully XLA-compatible.
            self.gdn = TPUGatedDeltaNet(
                hidden_size=d_model,
                num_heads=n_heads,
                **kwargs,
            )
            self._backend = "tpu"

        # ── Transformer block wrapper ─────────────────────────────────────
        # Pre-norm before GDN (attention sublayer)
        self.norm1 = nn.LayerNorm(d_model)

        # Pre-norm before FFN
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        ffn_dim = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x         : ``[B, T, d_model]``
            state     : Optional recurrent state for inference.
            use_cache : Whether to return the final recurrent state.

        Returns:
            ``(output [B, T, d_model], next_state or None)``
        """
        # ── Sublayer 1: Pre-Norm + GDN + Residual ─────────────────────────
        normed = self.norm1(x)
        res = self.gdn(normed, past_key_values=state, use_cache=use_cache)
        if isinstance(res, (tuple, list)):
            gdn_out = res[0]
            next_state = res[2] if len(res) > 2 else None
        else:
            gdn_out = res
            next_state = None
        x = x + gdn_out

        # ── Sublayer 2: Pre-Norm + FFN + Residual ─────────────────────────
        x = x + self.ffn(self.norm2(x))

        return x, next_state
