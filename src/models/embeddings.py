import torch
import torch.nn as nn
import torch.nn.functional as F


class GridEmbedding(nn.Module):
    """
    4×4 Patch Grid Embedding for ARC discrete grids.

    Paper spec (start_plan.md §3): "use 1×1 or 3×3 localized grid patches with
    categorical embeddings." We use 4×4 patches because:
      - 4 divides 64 exactly → no padding/masking complexity
      - Produces 16×16 = 256 patch tokens (+ CLS = 257 total)
      - Reduces self-attention from O(4097²) to O(257²) — 254× smaller

    Embedding strategy (Option A, paper-faithful):
      Each of the 16 cells in a 4×4 patch is embedded independently into d_model
      (preserving intra-patch spatial relationships), then the 16 embeddings are
      flattened and projected to a single d_model patch token via a linear layer.
    """

    PATCH_SIZE = 4  # fixed; changing this requires matching PositionalEncoding2D

    def __init__(self, d_model: int, num_colors: int = 16):
        super().__init__()
        self.d_model = d_model
        self.patch_size = self.PATCH_SIZE
        cells_per_patch = self.patch_size * self.patch_size  # 16

        # Per-cell color embedding (0-15 → d_model)
        self.cell_embed = nn.Embedding(num_colors, d_model)

        # Project 16 concatenated cell embeddings → 1 patch token
        self.patch_proj = nn.Linear(cells_per_patch * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., H, W] integer grid (H=W=64)
        Returns: [..., H//P, W//P, d_model]  (H//P = W//P = 16)
        """
        *batch_dims, H, W = x.shape
        P = self.patch_size
        Ph, Pw = H // P, W // P  # number of patches per axis (16 × 16)

        # Reshape into patches: [..., Ph, P, Pw, P]
        x_p = x.reshape(*batch_dims, Ph, P, Pw, P)
        # Permute to [..., Ph, Pw, P, P] — all cells of a patch together
        x_p = x_p.permute(*range(len(batch_dims)), -4, -2, -3, -1)
        # Flatten patch cells: [..., Ph, Pw, P*P]
        x_p = x_p.reshape(*batch_dims, Ph, Pw, P * P)

        # Embed each cell: [..., Ph, Pw, P*P, d_model]
        embedded = self.cell_embed(x_p)

        # Flatten cell embeddings and project to patch token: [..., Ph, Pw, d_model]
        patch_flat = embedded.reshape(*batch_dims, Ph, Pw, P * P * self.d_model)
        patch_tokens = self.patch_proj(patch_flat)  # [..., Ph, Pw, d_model]

        return patch_tokens


class ActionEmbedding(nn.Module):
    """
    Categorical Action Embedding: ActionType + X + Y.
    Implements z_a = ActionEmbed(9) + XEmbed(64) + YEmbed(64).
    """
    def __init__(self, d_model: int, num_actions: int = 9, grid_size: int = 64):
        super().__init__()
        self.action_type_embed = nn.Embedding(num_actions, d_model)
        self.x_embed = nn.Embedding(grid_size, d_model)
        self.y_embed = nn.Embedding(grid_size, d_model)

    def forward(self, action_type: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # action_type, x, y: [Batch] or [Batch, T]
        return self.action_type_embed(action_type) + self.x_embed(x) + self.y_embed(y)


class PositionalEncoding2D(nn.Module):
    """
    2D Absolute Positional Embeddings for the patch grid.

    Grid cells: 64×64 → Patch grid: 16×16 (for 4×4 patches).
    Positional embeddings operate at patch resolution.
    """
    def __init__(self, d_model: int, max_h: int = 16, max_w: int = 16):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, d_model)
        self.col_embed = nn.Embedding(max_w, d_model)

    def forward(self, h: int, w: int) -> torch.Tensor:
        """Returns [h, w, d_model] positional embeddings for an h×w patch grid."""
        rows = torch.arange(h, device=self.row_embed.weight.device)
        cols = torch.arange(w, device=self.col_embed.weight.device)

        row_emb = self.row_embed(rows).unsqueeze(1)  # [h, 1, d_model]
        col_emb = self.col_embed(cols).unsqueeze(0)  # [1, w, d_model]

        return row_emb + col_emb  # [h, w, d_model]
