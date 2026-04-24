import torch
import torch.nn as nn

class GridDecoder(nn.Module):
    """
    Decoder: Maps final latent state s_T back to the symbolic 64x64 grid.
    Output is logits over 16 colors.
    """
    def __init__(self, d_model: int, grid_size: int = 64, num_colors: int = 16):
        super().__init__()
        self.grid_size = grid_size
        self.num_colors = num_colors
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, grid_size * grid_size * num_colors)
        )

    def forward(self, s_T: torch.Tensor) -> torch.Tensor:
        """
        s_T: [Batch, d_model]
        Returns: [Batch, 16, 64, 64] logits
        """
        b = s_T.shape[0]
        logits = self.mlp(s_T)
        logits = logits.view(b, self.num_colors, self.grid_size, self.grid_size)
        return logits
