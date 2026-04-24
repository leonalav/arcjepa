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

    def forward_multistep(
        self,
        s_t: torch.Tensor,
        action_embeds: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """
        Multi-step rollout: predict k steps into the future without intermediate supervision.

        Args:
            s_t: [Batch, d_model] initial state latent
            action_embeds: [Batch, k, d_model] action embeddings for next k steps
            k: number of steps to predict

        Returns:
            predictions: [Batch, k, d_model] predicted latents at [t+1, t+2, ..., t+k]
        """
        predictions = []
        s_curr = s_t  # [Batch, d_model]

        for i in range(k):
            # Get action embedding for this step
            z_a = action_embeds[:, i, :]  # [Batch, d_model]

            # Predict next state (single-step forward)
            # Need to add time dimension for forward compatibility
            s_curr_expanded = s_curr.unsqueeze(1)  # [Batch, 1, d_model]
            z_a_expanded = z_a.unsqueeze(1)  # [Batch, 1, d_model]

            s_next = self.forward(s_curr_expanded, z_a_expanded)  # [Batch, 1, d_model]
            s_next = s_next.squeeze(1)  # [Batch, d_model]

            predictions.append(s_next)
            s_curr = s_next  # Update current state for next iteration

        # Stack predictions: [Batch, k, d_model]
        return torch.stack(predictions, dim=1)
