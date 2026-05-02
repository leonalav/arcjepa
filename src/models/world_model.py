"""
ARC-JEPA World Model — game-aware V-JEPA-style architecture for ARC-AGI-3.
"""

import copy
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.arc_schema import DEFAULT_NUM_ACTIONS, masked_action_logits
from .decoder import GridDecoder
from .embeddings import ActionEmbedding, GameEmbedding, GridEmbedding, MetadataEmbedding, PositionalEncoding2D
from .jepa_predictor import JEPAPredictor
from .sequence_model import GDNSequenceModel, GDNState
from .spatial_encoder import DiscreteViT


class PolicyHead(nn.Module):
    def __init__(self, d_model: int, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.action = nn.Linear(d_model, num_actions)
        self.x = nn.Linear(d_model, 64)
        self.y = nn.Linear(d_model, 64)

    def components(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(latents)
        return self.action(h), self.x(h), self.y(h)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.components(latents), dim=-1)


class ARCJEPAWorldModel(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        num_vit_layers: int = 4,
        num_gdn_heads: int = 4,
        tau: float = 0.999,
        multistep_k: int = 1,
        predictor_layers: int = 6,
        predictor_bottleneck: int = 384,
        num_actions: int = DEFAULT_NUM_ACTIONS,
        max_games: int = 4096,
        max_game_families: int = 512,
        use_game_conditioning: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.multistep_k = multistep_k
        self.num_actions = num_actions
        self.max_games = max_games
        self.max_game_families = max_game_families
        self.use_game_conditioning = use_game_conditioning

        self.grid_embed = GridEmbedding(d_model)
        self.pos_embed = PositionalEncoding2D(d_model)
        self.action_embed = ActionEmbedding(d_model, num_actions=num_actions)
        self.game_embed = GameEmbedding(d_model, max_games=max_games, max_game_families=max_game_families)
        self.metadata_embed = MetadataEmbedding(d_model)

        self.online_encoder = DiscreteViT(d_model, nhead=n_heads, num_layers=num_vit_layers)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.gdn = GDNSequenceModel(d_model, n_heads=num_gdn_heads)
        self.predictor = JEPAPredictor(
            d_model=d_model,
            num_layers=predictor_layers,
            bottleneck_dim=predictor_bottleneck,
            num_heads=max(1, predictor_bottleneck // 32),
        )
        self.decoder = GridDecoder(d_model)
        self.policy_head = PolicyHead(d_model, num_actions)
        self.terminal_head = nn.Linear(d_model, 1)
        self.value_head = nn.Linear(d_model, 1)
        self.efficiency_head = nn.Linear(d_model, 1)

    def encode(self, grids: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        b, t, h, w = grids.shape
        grids_flat = grids.reshape(b * t, h, w)
        x = self.grid_embed(grids_flat)
        Ph, Pw = x.shape[1], x.shape[2]
        pos = self.pos_embed(Ph, Pw)
        x = x + pos.unsqueeze(0)
        latents = encoder(x)
        return latents.reshape(b, t, self.d_model)

    def _default_temporal(self, batch: Dict[str, torch.Tensor], key: str, shape: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
        states = batch['states']
        return torch.zeros(shape, dtype=dtype, device=states.device)

    def _conditioning(self, batch: Dict[str, torch.Tensor], length: int) -> torch.Tensor:
        states = batch['states']
        B = states.shape[0]
        device = states.device
        game_id = batch.get('game_id', torch.zeros(B, states.shape[1], dtype=torch.long, device=device))[:, :length]
        game_family = batch.get('game_family', torch.zeros(B, states.shape[1], dtype=torch.long, device=device))[:, :length]
        terminal = batch.get('terminal', torch.zeros(B, states.shape[1], dtype=torch.float32, device=device))[:, :length]
        success = batch.get('success', torch.zeros(B, states.shape[1], dtype=torch.float32, device=device))[:, :length]
        score = batch.get('score', torch.zeros(B, states.shape[1], dtype=torch.float32, device=device))[:, :length]
        step_index = batch.get('step_index', torch.zeros(B, states.shape[1], dtype=torch.long, device=device))[:, :length]

        cond = self.metadata_embed(terminal, success, score, step_index)
        if self.use_game_conditioning:
            cond = cond + self.game_embed(game_id, game_family)
        return cond

    def _split_heads(self, pred_latents: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        action_logits, x_logits, y_logits = self.policy_head.components(pred_latents)
        masked_logits = masked_action_logits(action_logits, action_mask)
        return {
            'action_logits': masked_logits,
            'raw_action_logits': action_logits,
            'x_logits': x_logits,
            'y_logits': y_logits,
            'policy_logits': torch.cat([masked_logits, x_logits, y_logits], dim=-1),
            'terminal_logits': self.terminal_head(pred_latents).squeeze(-1),
            'value_pred': self.value_head(pred_latents).squeeze(-1),
            'efficiency_pred': self.efficiency_head(pred_latents).squeeze(-1),
        }

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        context_ratio: float = 0.7,
        use_teacher_forcing: bool = True
    ) -> Dict[str, torch.Tensor]:
        states = batch['states']
        actions = batch['actions'].clamp(0, self.num_actions - 1)
        cx, cy = batch['coords_x'], batch['coords_y']
        target_grids = batch['target_states']

        B, T = states.shape[:2]
        K = max(1, int(T * context_ratio))
        all_online_latents = self.encode(states[:, :T-1], self.online_encoder)
        all_online_latents = all_online_latents + self._conditioning(batch, T - 1)

        context_latents = all_online_latents[:, :K]
        context_features, _ = self.gdn(context_latents, use_cache=True)

        if use_teacher_forcing:
            action_embeds = self.action_embed(actions[:, K-1:T-1], cx[:, K-1:T-1], cy[:, K-1:T-1])
            action_embeds = action_embeds + self._conditioning(batch, T - 1)[:, K-1:T-1]
            pred_latents = self.predictor(context_features, action_embeds)
            if self.multistep_k > 1 and K-1 + self.multistep_k <= T:
                s_init = all_online_latents[:, K-1]
                action_embeds_k = self.action_embed(
                    actions[:, K-1 : K-1+self.multistep_k],
                    cx[:, K-1 : K-1+self.multistep_k],
                    cy[:, K-1 : K-1+self.multistep_k]
                )
                action_embeds_k = action_embeds_k + self._conditioning(batch, T)[:, K-1 : K-1+self.multistep_k]
                multistep_pred_latents = self.predictor.forward_multistep(s_init, action_embeds_k, self.multistep_k)
            else:
                multistep_pred_latents = None
        else:
            pred_latents = []
            context_for_ar = context_features
            cond = self._conditioning(batch, T)
            for t in range(K-1, T-1):
                z_a = self.action_embed(actions[:, t], cx[:, t], cy[:, t]).unsqueeze(1) + cond[:, t:t+1]
                s_next_pred = self.predictor(context_for_ar, z_a).squeeze(1)
                pred_latents.append(s_next_pred)
                context_for_ar = torch.cat([context_for_ar, s_next_pred.unsqueeze(1)], dim=1)
            pred_latents = torch.stack(pred_latents, dim=1)
            multistep_pred_latents = None

        with torch.no_grad():
            s_next_target = self.encode(target_grids[:, K-1:T-1], self.target_encoder)
            s_next_target = F.layer_norm(s_next_target, (s_next_target.size(-1),))
            if self.multistep_k > 1 and K - 1 + self.multistep_k <= T:
                multistep_target_latents = self.encode(target_grids[:, K-1 : K-1+self.multistep_k], self.target_encoder)
                multistep_target_latents = F.layer_norm(multistep_target_latents, (multistep_target_latents.size(-1),))
            else:
                multistep_target_latents = None

        final_state_logits = self.decoder(pred_latents[:, -1])
        action_mask = batch.get('available_actions_mask', None)
        if action_mask is not None:
            action_mask = action_mask[:, K:T]
            if action_mask.shape[1] != pred_latents.shape[1]:
                action_mask = action_mask[:, :pred_latents.shape[1]]

        output = {
            'pred_latents': pred_latents,
            'target_latents': s_next_target,
            'decoder_logits': final_state_logits,
            **self._split_heads(pred_latents, action_mask),
        }
        if multistep_pred_latents is not None:
            output['multistep_pred_latents'] = multistep_pred_latents
            output['multistep_target_latents'] = multistep_target_latents
        return output

    def init_inference(
        self,
        context_grids: torch.Tensor,
        game_id: Optional[torch.Tensor] = None,
        game_family: Optional[torch.Tensor] = None,
    ) -> 'InferenceState':
        ctx_latents = self.encode(context_grids, self.online_encoder)
        if game_id is not None and game_family is not None:
            ctx_latents = ctx_latents + self.game_embed(game_id, game_family)
        _, gdn_state = self.gdn(ctx_latents, use_cache=True)
        return InferenceState(latent=ctx_latents[:, -1], gdn_state=gdn_state, step_idx=0)

    def inference_step(
        self,
        state: 'InferenceState',
        action: torch.Tensor,
        coord_x: torch.Tensor,
        coord_y: torch.Tensor,
        available_actions_mask: Optional[torch.Tensor] = None,
        game_id: Optional[torch.Tensor] = None,
        game_family: Optional[torch.Tensor] = None,
    ) -> Tuple['InferenceState', Dict[str, torch.Tensor]]:
        z_a = self.action_embed(action.unsqueeze(1), coord_x.unsqueeze(1), coord_y.unsqueeze(1)).squeeze(1)
        if game_id is not None and game_family is not None:
            z_a = z_a + self.game_embed(game_id, game_family)
        gdn_summary, next_gdn_state = self.gdn.step(state.latent.unsqueeze(1), state.gdn_state)
        gdn_summary = gdn_summary.squeeze(1)
        next_latent = self.predictor.forward_step(
            s_t=state.latent,
            action_embed=z_a,
            gdn_state_summary=gdn_summary,
            step_idx=state.step_idx
        )
        decoder_logits = self.decoder(next_latent)
        outputs = {
            **self._split_heads(next_latent.unsqueeze(1), available_actions_mask.unsqueeze(1) if available_actions_mask is not None else None),
            'decoder_logits': decoder_logits,
            'pred_latent': next_latent,
        }
        next_state = InferenceState(latent=next_latent, gdn_state=next_gdn_state, step_idx=state.step_idx + 1)
        return next_state, outputs


class InferenceState(NamedTuple):
    latent: torch.Tensor
    gdn_state: GDNState
    step_idx: int
