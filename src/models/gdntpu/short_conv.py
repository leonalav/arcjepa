# gdntpu/short_conv.py
# TPU-compatible depthwise causal convolution.
# Replaces fla.modules.ShortConvolution (which dispatches to Triton/CUDA).
# State-dict shape match: weight [C, 1, W], bias [C] — identical to FLA.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PureShortConvolution(nn.Conv1d):
    """Depthwise causal 1-D convolution without any Triton / CUDA kernels.

    Inherits ``nn.Conv1d`` so that ``state_dict()`` keys and parameter shapes
    are **byte-for-byte identical** to FLA's ``ShortConvolution``:

        weight : [hidden_size, 1, kernel_size]
        bias   : [hidden_size]   (or absent when bias=False)

    Causal padding is achieved by prepending ``kernel_size - 1`` zeros to the
    time dimension *before* the convolution, then slicing back to length ``T``.
    This is equivalent to the left-padding that FLA's Triton kernel performs.

    Args:
        hidden_size  : Number of channels (= groups, depthwise conv).
        kernel_size  : Convolution kernel width.
        bias         : Whether to include a bias term.
        activation   : ``'silu'`` or ``None``.  Matches FLA's ``activation``
                       argument; only ``'silu'`` / ``'swish'`` are supported.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = "silu",
    ) -> None:
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            # nn.Conv1d needs *some* padding value at construction; we override
            # it with manual padding in forward so this is irrelevant.
            padding=0,
        )
        self.hidden_size = hidden_size
        if activation is not None and activation not in ("silu", "swish"):
            raise ValueError(
                f"PureShortConvolution: unsupported activation '{activation}'. "
                "Only 'silu'/'swish' are supported."
            )
        self.activation = activation

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x : ``[B, T, C]`` — input tensor.
            cache : ignored during training; kept for API parity with FLA.
            output_final_state : ignored during training.

        Returns:
            ``(output, None)`` where output has shape ``[B, T, C]``.
        """
        B, T, C = x.shape

        # [B, T, C] → [B, C, T]  (Conv1d expects channel-first)
        x_t = x.transpose(1, 2)

        # Causal padding: prepend (kernel_size-1) zeros on the left of T.
        pad_len = self.kernel_size[0] - 1
        x_t = F.pad(x_t, (pad_len, 0))

        # Depthwise 1-D convolution (no extra padding — we did it manually).
        out = F.conv1d(
            x_t,
            self.weight,
            self.bias,
            stride=1,
            padding=0,
            groups=self.groups,
        )

        # out shape is [B, C, T] — slice to length T (no right-leakage).
        out = out[..., :T]

        # [B, C, T] → [B, T, C]
        out = out.transpose(1, 2)

        if self.activation is not None:
            out = F.silu(out)

        return out, None
