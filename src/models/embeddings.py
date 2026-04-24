import torch
import torch.nn as nn

class GridEmbedding(nn.Module):
    """Maps discrete color integers (0-15) to d_model vectors."""
    def __init__(self, d_model: int, num_colors: int = 16): # max_val=16 (0-15)
        super().__init__()
        self.embedding = nn.Embedding(num_colors, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., H, W]
        return self.embedding(x)

class ActionEmbedding(nn.Module):
    """
    Categorical Action Embedding: ActionType + X + Y.
    Implements the z_a = ActionEmbed(7) + XEmbed(30) + YEmbed(30) logic.
    """
    def __init__(self, d_model: int, num_actions: int = 8, grid_size: int = 64):
        super().__init__()
        # 0 is NONE/PAD, 1-7 are ACTION1-7
        self.action_type_embed = nn.Embedding(num_actions, d_model)
        # 0-63 for coordinates (max 64x64 grid)
        self.x_embed = nn.Embedding(grid_size, d_model)
        self.y_embed = nn.Embedding(grid_size, d_model)

    def forward(self, action_type: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # action_type, x, y: [Batch, T]
        type_emb = self.action_type_embed(action_type)
        x_emb = self.x_embed(x)
        y_emb = self.y_embed(y)
        return type_emb + x_emb + y_emb

class PositionalEncoding2D(nn.Module):
    """2D Absolute Positional Embeddings for HxW grid."""
    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, d_model)
        self.col_embed = nn.Embedding(max_w, d_model)

    def forward(self, h: int, w: int) -> torch.Tensor:
        # Returns [h, w, d_model]
        rows = torch.arange(h, device=self.row_embed.weight.device)
        cols = torch.arange(w, device=self.col_embed.weight.device)
        
        row_emb = self.row_embed(rows).unsqueeze(1) # [h, 1, d_model]
        col_emb = self.col_embed(cols).unsqueeze(0) # [1, w, d_model]
        
        return row_emb + col_emb
