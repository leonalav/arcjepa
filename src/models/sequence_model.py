import torch
import torch.nn as nn
from typing import Optional, Tuple, NamedTuple

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


class GDNState(NamedTuple):
    """Fixed-size recurrent state for O(1) inference.

    Memory footprint: B × HV × K × V × sizeof(float32)
    With default dims (HV=8, K=64, V=128): 8×64×128×4 = 256 KB per sample.
    This is INDEPENDENT of sequence length — the entire history is compressed
    into this fixed-size matrix via the delta rule's associative updates.
    """
    gdn_state: torch.Tensor  # [B, HV, K, V] — GDN recurrent state


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

    Inference complexity guarantees:
    - ``forward()``: O(N) time, O(N) memory — processes full sequence
    - ``step()``:    O(1) time, O(1) memory — single-token recurrent step
      The recurrent state S ∈ ℝ^{H×K×V} is a fixed-size matrix that
      compresses the entire history via the delta rule's associative updates.
      MCTS rollouts use ``step()`` for O(1) per move.

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
        self.d_model = d_model

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
        state: Optional[GDNState] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[GDNState]]:
        """
        Full-sequence forward pass. O(N) time, O(N) memory.

        Used during:
        - Training (teacher forcing)
        - Initial context processing before MCTS rollout

        Args:
            x         : ``[B, T, d_model]``
            state     : Optional prior ``GDNState`` (for continuation).
            use_cache : If True, return the final recurrent state for
                        subsequent ``step()`` calls.

        Returns:
            ``(output [B, T, d_model], GDNState or None)``
        """
        gdn_state = state.gdn_state if state is not None else None

        # ── Sublayer 1: Pre-Norm + GDN + Residual ─────────────────────────
        normed = self.norm1(x)
        res = self.gdn(normed, past_key_values=gdn_state, use_cache=use_cache)
        if isinstance(res, (tuple, list)):
            gdn_out = res[0]
            next_gdn_state = res[2] if len(res) > 2 else None
        else:
            gdn_out = res
            next_gdn_state = None
        x = x + gdn_out

        # ── Sublayer 2: Pre-Norm + FFN + Residual ─────────────────────────
        x = x + self.ffn(self.norm2(x))

        next_state = GDNState(gdn_state=next_gdn_state) if next_gdn_state is not None else None
        return x, next_state

    def step(
        self,
        x_t: torch.Tensor,
        state: GDNState,
    ) -> Tuple[torch.Tensor, GDNState]:
        """
        Single-token recurrent step. **O(1) time, O(1) memory.**

        This is the method MCTS rollouts should call. The recurrent state
        ``S ∈ ℝ^{B × HV × K × V}`` is a fixed-size matrix — its size is
        independent of how many prior tokens have been processed.

        Mathematical guarantee (GDN paper, Eq. 10):
            S_t = α_t (I − β_t k_t k_t^T) S_{t-1} + β_t v_t k_t^T
            o_t = S_t q_t
        Each step applies a rank-1 update to S: O(K×V) = O(1) per step.

        Args:
            x_t   : ``[B, 1, d_model]`` — single input token.
            state : ``GDNState`` from prior ``forward()`` or ``step()`` call.

        Returns:
            ``(output [B, 1, d_model], next_state: GDNState)``
        """
        # ── Sublayer 1: Pre-Norm + GDN (recurrent, T=1) + Residual ────────
        normed = self.norm1(x_t)
        # GDN with T=1 → recurrent mode → single step, O(1) memory
        res = self.gdn(
            normed,
            past_key_values=state.gdn_state,
            use_cache=True  # always output final state for next step
        )
        if isinstance(res, (tuple, list)):
            gdn_out = res[0]
            next_gdn_state = res[2] if len(res) > 2 else state.gdn_state
        else:
            gdn_out = res
            next_gdn_state = state.gdn_state

        x_t = x_t + gdn_out

        # ── Sublayer 2: Pre-Norm + FFN + Residual ─────────────────────────
        x_t = x_t + self.ffn(self.norm2(x_t))

        return x_t, GDNState(gdn_state=next_gdn_state)
