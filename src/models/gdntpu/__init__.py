# gdntpu/__init__.py
# Package init for the TPU-compatible Gated DeltaNet implementation.
#
# Usage:
#   from src.models.gdntpu import GatedDeltaNet

from .gated_deltanet import GatedDeltaNet
from .short_conv import PureShortConvolution
from .norm import PureRMSNorm, PureRMSNormGated
from .delta_rule import (
    pure_chunk_gated_delta_rule,
    pure_recurrent_gated_delta_rule,
    _compute_gate,
)

__all__ = [
    "GatedDeltaNet",
    "PureShortConvolution",
    "PureRMSNorm",
    "PureRMSNormGated",
    "pure_chunk_gated_delta_rule",
    "pure_recurrent_gated_delta_rule",
    "_compute_gate",
]
