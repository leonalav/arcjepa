# gdntpu/norm.py
# TPU-compatible RMSNorm and RMSNormGated.
# Replaces fla.modules.FusedRMSNormGated and fla.modules.RMSNorm.
# State-dict shape match: weight [hidden_size] — identical to FLA.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pure RMSNorm
# ---------------------------------------------------------------------------
class PureRMSNorm(nn.Module):
    """Root-mean-square normalisation with a learnable per-channel scale.

    Computes::

        y = x / sqrt(mean(x², dim=-1, keepdim=True) + eps) * weight

    This is mathematically identical to FLA's ``RMSNorm`` (which calls the
    same Triton ``layer_norm`` kernel with ``is_rms_norm=True``).

    State-dict keys:
        weight : ``[hidden_size]``, initialised to ones.
        (bias is registered as None to match FLA.)

    Args:
        hidden_size : Feature dimension to normalise.
        eps         : Stability epsilon (default 1e-5, matching FLA).
        dtype       : Optional forced parameter dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        dtype: torch.dtype | None = torch.float32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        # FLA's RMSNorm always has an elementwise_affine weight and no bias.
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : ``[..., hidden_size]``
        Returns:
            Normalised tensor of same shape.
        """
        # Cast to float32 for numerical stability (matches FLA kernel).
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        # Apply weight; cast result back to input dtype.
        return (x_norm * self.weight.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# Pure RMSNormGated  (replaces FusedRMSNormGated with activation='swish')
# ---------------------------------------------------------------------------
class PureRMSNormGated(nn.Module):
    """RMS normalisation followed by a SiLU-gated multiply.

    Computes::

        y = RMSNorm(x) * silu(g)

    This is the exact floating-point equivalent of FLA's
    ``FusedRMSNormGated`` Triton kernel (``fused_norm_gate.py`` lines 97-98):

        b_y = b_y * b_g * tl.sigmoid(b_g)   # swish / silu path

    State-dict keys (matching FLA ``FusedRMSNormGated``):
        weight : ``[hidden_size]``, init ones.
        (bias registered as None.)

    Args:
        hidden_size : Dimension of ``x`` and ``g``.
        eps         : Epsilon for RMS normalisation (default 1e-5).
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # FLA registers bias as None for FusedRMSNormGated.
        self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x : ``[..., hidden_size]`` — hidden states to normalise.
            g : ``[..., hidden_size]`` — gate tensor (SiLU is applied here).
        Returns:
            Tensor of shape ``[..., hidden_size]``.
        """
        # RMS normalise x in float32.
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        x_norm = x_norm * self.weight.float()

        # Gate: silu(g) = g * sigmoid(g).
        gate = F.silu(g.float())

        # Fused multiply, cast back to original dtype.
        return (x_norm * gate).to(x.dtype)
