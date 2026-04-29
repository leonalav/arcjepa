# gdntpu/gated_deltanet.py
# Pure-PyTorch / XLA-compatible GatedDeltaNet layer.
#
# This file is a 1:1 mathematical and state-dict equivalent of:
#   fla.layers.GatedDeltaNet  (support_files/gated_deltanet.py)
#
# Every parameter name, shape, and initialization procedure is preserved so
# that checkpoints can be loaded interchangeably between this TPU implementation
# and the FLA Triton-based GPU implementation.
#
# Removed FLA-only imports:
#   - fla.layers.utils (get_layer_cache, get_unpad_data, …)  — varlen not needed
#   - fla.modules.FusedRMSNormGated, RMSNorm, ShortConvolution — replaced below
#   - fla.ops.gated_delta_rule                                 — replaced below
#
# Paper: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule"
#        https://arxiv.org/abs/2412.06464

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from .short_conv import PureShortConvolution
from .norm import PureRMSNorm, PureRMSNormGated
from .delta_rule import (
    _compute_gate,
    pure_chunk_gated_delta_rule,
    pure_recurrent_gated_delta_rule,
)


class GatedDeltaNet(nn.Module):
    """TPU-friendly Gated Delta Network layer.

    Drop-in replacement for ``fla.layers.GatedDeltaNet``.  All parameter names,
    shapes, and initialisation procedures are identical, guaranteeing that
    ``state_dict()`` keys are compatible between this module and the FLA
    Triton implementation.

    Mathematical reference (paper Eq. 10):
        S_t = α_t (I − β_t k_t k_t^T) S_{t-1} + β_t v_t k_t^T
        o_t = S_t q_t
    where α_t = exp(g_t), g_t = −exp(A_log) ⊙ softplus(a_proj(x) + dt_bias).

    Args match FLA's GatedDeltaNet exactly (see support_files/gated_deltanet.py).
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int | None = None,
        mode: str = "chunk",
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()

        # ---- Config --------------------------------------------------------
        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim   = int(self.num_heads   * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # Validation (mirrors FLA gated_deltanet.py lines 127-142).
        if not math.isclose(self.num_v_heads * head_dim * expand_v,
                            self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value_dim. "
                f"Got {self.num_v_heads * head_dim * expand_v}."
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by "
                f"num_heads={self.num_heads}."
            )
        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer head_v_dim. "
                f"Got {head_dim * expand_v}."
            )
        assert mode in ("chunk", "fused_recurrent"), (
            f"Unsupported mode '{mode}'."
        )

        # ---- Linear projections (FLA lines 144-148) ----------------------
        self.q_proj = nn.Linear(hidden_size, self.key_dim,   bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim,   bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)  # gate input
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)  # beta

        # ---- SSM-style decay parameters (FLA lines 150-167) -------------
        # A_log: log of a uniform sample from (0, 16). Shape [num_v_heads].
        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # dt_bias: inverse-softplus of a log-uniform dt in [dt_min, dt_max].
        dt_min, dt_max = 0.001, 0.1
        dt_init_floor  = 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]

        # ---- Short convolutions (FLA lines 169-193) ----------------------
        if use_short_conv:
            self.q_conv1d = PureShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = PureShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = PureShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to GDN performance. "
                "Only disable if you know what you are doing.",
            )

        # ---- Output gate & norm (FLA lines 194-199) ----------------------
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = PureRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = PureRMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, None]:
        """
        Args:
            hidden_states : ``[B, T, hidden_size]``

        Returns:
            ``(output, None, None)`` — matching FLA's return signature.
        """
        batch_size, q_len, _ = hidden_states.shape

        # During training always use chunk mode (FLA lines 219-221).
        if q_len <= 64:
            mode = "fused_recurrent"
        else:
            mode = "chunk" if self.training else self.mode

        # ---- Project and (optionally) short-conv q, k, v ---------------
        if self.use_short_conv:
            q, _ = self.q_conv1d(self.q_proj(hidden_states))   # [B, T, key_dim]
            k, _ = self.k_conv1d(self.k_proj(hidden_states))
            v, _ = self.v_conv1d(self.v_proj(hidden_states))   # [B, T, value_dim]
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        # Reshape to multi-head (FLA lines 257-258).
        q = rearrange(q, "b t (h d) -> b t h d", d=self.head_k_dim)   # [B,T,H,K]
        k = rearrange(k, "b t (h d) -> b t h d", d=self.head_k_dim)
        v = rearrange(v, "b t (h d) -> b t h d", d=self.head_v_dim)   # [B,T,HV,V]

        # ---- Beta (write-strength) (FLA lines 260-262) ------------------
        beta = self.b_proj(hidden_states).sigmoid()   # [B, T, HV]
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- Gate: raw → log-space decay (FLA lines 266-279) -----------
        # a_proj outputs raw g; gate transform: g = −exp(A_log)·softplus(g+dt_bias)
        g_raw = self.a_proj(hidden_states)             # [B, T, HV]
        g_log = _compute_gate(g_raw, self.A_log, self.dt_bias)  # [B, T, HV]

        # ---- L2-normalise q and k (use_qk_l2norm_in_kernel=True) -------
        # FLA fused_recurrent kernel lines 107-108; eps=1e-6 from l2norm.py:41.
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)

        # Scale factor (applied to q inside the kernel: naive.py line 48).
        scale = self.head_k_dim ** -0.5

        # ---- Run delta rule (kernel dispatch) ---------------------------
        if mode == "chunk":
            o, _ = pure_chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g_log,
                beta=beta,
                scale=scale,
                chunk_size=64,
                initial_state=None,
                output_final_state=False,
            )
        else:  # fused_recurrent
            o, _ = pure_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g_log,
                beta=beta,
                scale=scale,
                initial_state=None,
                output_final_state=False,
            )
        # o: [B, T, HV, V]

        # ---- Output norm + gate (FLA lines 306-311) --------------------
        if self.use_gate:
            g_out = rearrange(
                self.g_proj(hidden_states), "b t (h d) -> b t h d", d=self.head_v_dim
            )  # [B, T, HV, V]
            o = self.o_norm(o, g_out)
        else:
            o = self.o_norm(o)

        # ---- Flatten heads and project (FLA lines 311-312) --------------
        o = rearrange(o, "b t h d -> b t (h d)")   # [B, T, value_dim]
        o = self.o_proj(o)                           # [B, T, hidden_size]

        return o, None, past_key_values
