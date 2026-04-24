import torch
import torch.nn as nn
from typing import Optional

class JEPAPredictor(nn.Module):
    """
    Predictor Module: Maps current state s_t and action z_a to next state s_{t+1}.
    Implements the z_a + s_t -> s_{t+1} logic.
    """
    def __init__(self, d_model: int, num_layers: int = 2, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model * 2
            
        layers = []
        curr_dim = d_model
        for i in range(num_layers):
            layers.extend([
                nn.Linear(curr_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, d_model)
            ])
            curr_dim = d_model
            
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, s_t: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """
        s_t: [Batch, T, d_model]
        z_a: [Batch, T, d_model]
        Returns: [Batch, T, d_model] predicted s_{t+1}
        """
        # We use additive conditioning as suggested
        x = s_t + z_a
        
        # Apply MLP to predict the delta or the absolute next state
        # Here we predict the next state directly
        s_next_pred = self.mlp(x)
        
        # Add residual connection from s_t
        return self.norm(s_next_pred + s_t)
