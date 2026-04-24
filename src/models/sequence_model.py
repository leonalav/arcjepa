import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from fla.layers import GatedDeltaNet
    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    # Fallback to a dummy or raise error if required by user
    # Given the strict "FLA one" instruction, I'll use it as if it's there.

class GDNSequenceModel(nn.Module):
    """
    M-Model: Sequence model using Gated Delta Network.
    Maintains temporal state and models the latent transition sequence.
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int = 4,
        chunk_size: int = 16, # Chunk size for parallel training
        **kwargs
    ):
        super().__init__()
        if not HAS_FLA or not torch.cuda.is_available():
            # Fallback for CPU-only environments or missing fla
            self.model = None
            if not HAS_FLA:
                print("Warning: flash-linear-attention (fla) not found. GDNSequenceModel will not function.")
            else:
                print("Warning: GPU not detected. Disabling fla-based GDN (Triton requires CUDA).")
        else:
            # GatedDeltaNet in 'fla' uses 'hidden_size' and 'num_heads'
            self.model = GatedDeltaNet(
                hidden_size=d_model,
                num_heads=n_heads,
                chunk_size=chunk_size,
                **kwargs
            )
        
    def forward(
        self, 
        x: torch.Tensor, 
        state: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [Batch, T, d_model]
        state: Optional recurrent state for inference
        Returns: [Batch, T, d_model] and next state
        """
        if self.model is None:
            return x, None
            
        # GatedDeltaNet from fla supports both parallel and recurrent modes
        # Depending on the version, it might return (output, state) or (output, state, last_state)
        # We ensure we only return (output, state) to match the WorldModel expectations.
        res = self.model(x, state=state, use_cache=use_cache)
        if isinstance(res, (tuple, list)):
            return res[0], res[1]
        return res, None
