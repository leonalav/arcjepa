import torch
import torch.nn as nn


class GridEmbedding(nn.Module):
    PATCH_SIZE = 4

    def __init__(self, d_model: int, num_colors: int = 16):
        super().__init__()
        self.d_model = d_model
        self.patch_size = self.PATCH_SIZE
        cells_per_patch = self.patch_size * self.patch_size
        self.cell_embed = nn.Embedding(num_colors, d_model)
        self.patch_proj = nn.Linear(cells_per_patch * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch_dims, H, W = x.shape
        P = self.patch_size
        Ph, Pw = H // P, W // P
        x_p = x.reshape(*batch_dims, Ph, P, Pw, P)
        x_p = x_p.permute(*range(len(batch_dims)), -4, -2, -3, -1)
        x_p = x_p.reshape(*batch_dims, Ph, Pw, P * P)
        embedded = self.cell_embed(x_p)
        patch_flat = embedded.reshape(*batch_dims, Ph, Pw, P * P * self.d_model)
        return self.patch_proj(patch_flat)


class ActionEmbedding(nn.Module):
    def __init__(self, d_model: int, num_actions: int = 10, grid_size: int = 64):
        super().__init__()
        self.num_actions = num_actions
        self.grid_size = grid_size
        self.action_type_embed = nn.Embedding(num_actions, d_model)
        self.x_embed = nn.Embedding(grid_size, d_model)
        self.y_embed = nn.Embedding(grid_size, d_model)

    def forward(self, action_type: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        action_type = action_type.clamp(0, self.num_actions - 1)
        x = x.clamp(0, self.grid_size - 1)
        y = y.clamp(0, self.grid_size - 1)
        return self.action_type_embed(action_type) + self.x_embed(x) + self.y_embed(y)


class GameEmbedding(nn.Module):
    def __init__(self, d_model: int, max_games: int = 4096, max_game_families: int = 512):
        super().__init__()
        self.max_games = max_games
        self.max_game_families = max_game_families
        self.game_embed = nn.Embedding(max_games, d_model)
        self.family_embed = nn.Embedding(max_game_families, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, game_id: torch.Tensor, game_family: torch.Tensor) -> torch.Tensor:
        game_id = game_id.clamp(0, self.max_games - 1)
        game_family = game_family.clamp(0, self.max_game_families - 1)
        return self.norm(self.game_embed(game_id) + self.family_embed(game_family))


class MetadataEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        terminal: torch.Tensor,
        success: torch.Tensor,
        score: torch.Tensor,
        step_index: torch.Tensor,
    ) -> torch.Tensor:
        step_scaled = step_index.float() / 100.0
        features = torch.stack([
            terminal.float(),
            success.float(),
            score.float(),
            step_scaled,
        ], dim=-1)
        return self.proj(features)


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, max_h: int = 16, max_w: int = 16):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, d_model)
        self.col_embed = nn.Embedding(max_w, d_model)

    def forward(self, h: int, w: int) -> torch.Tensor:
        rows = torch.arange(h, device=self.row_embed.weight.device)
        cols = torch.arange(w, device=self.col_embed.weight.device)
        row_emb = self.row_embed(rows).unsqueeze(1)
        col_emb = self.col_embed(cols).unsqueeze(0)
        return row_emb + col_emb
