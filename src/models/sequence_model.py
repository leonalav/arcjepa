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
    """Sequence model using Gated Delta Network.

    On TPU/CPU: uses the pure-PyTorch ``gdntpu.GatedDeltaNet`` (XLA-safe).
    On CUDA (with fla installed): uses ``fla.layers.GatedDeltaNet`` (Triton).

    State-dict keys are identical between both backends, so checkpoints saved
    on TPU can be loaded directly into the CUDA variant and vice-versa.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        **kwargs,
    ):
        super().__init__()

        is_cuda = torch.cuda.is_available()

        if is_cuda and HAS_FLA:
            # Fast GPU path via Triton kernels.
            self.model = FLAGatedDeltaNet(
                hidden_size=d_model,
                num_heads=n_heads,
                **kwargs,
            )
            self._backend = "fla"
        else:
            # TPU / CPU path — pure PyTorch, fully XLA-compatible.
            self.model = TPUGatedDeltaNet(
                hidden_size=d_model,
                num_heads=n_heads,
                **kwargs,
            )
            self._backend = "tpu"

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
        # Both backends return (output, attn_weights, past_key_values).
        res = self.model(x, past_key_values=state, use_cache=use_cache)
        if isinstance(res, (tuple, list)):
            return res[0], res[2]   # output, past_key_values
        return res, None
