from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.arc_schema import DEFAULT_NUM_ACTIONS, masked_action_logits, stable_game_family_id, stable_game_id
from src.env.types import ARCAction
from src.models.world_model import ARCJEPAWorldModel
from .base import ActionDistribution, Agent


class ModelAgent(Agent):
    def __init__(self, checkpoint: str | Path | None = None, device: str | torch.device | None = None, num_actions: int = DEFAULT_NUM_ACTIONS):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_actions = num_actions
        self.model = ARCJEPAWorldModel(num_actions=num_actions).to(self.device)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def propose(self, obs, legal_actions: list[ARCAction], budget=None) -> ActionDistribution:
        if not legal_actions:
            return ActionDistribution([], [])
        with torch.no_grad():
            grid = torch.tensor(obs.grid, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(0)
            batch = {
                "states": grid.repeat(1, 2, 1, 1),
                "target_states": grid.repeat(1, 2, 1, 1),
                "actions": torch.zeros(1, 2, dtype=torch.long, device=self.device),
                "coords_x": torch.zeros(1, 2, dtype=torch.long, device=self.device),
                "coords_y": torch.zeros(1, 2, dtype=torch.long, device=self.device),
                "game_id": torch.full((1, 2), stable_game_id(obs.game_id), dtype=torch.long, device=self.device),
                "game_family": torch.full((1, 2), stable_game_family_id(obs.game_id), dtype=torch.long, device=self.device),
            }
            outputs = self.model(batch, context_ratio=0.5)
            logits = outputs["raw_action_logits"][:, -1]
            mask = torch.zeros(1, self.num_actions, dtype=torch.bool, device=self.device)
            for action in legal_actions:
                if 0 <= action.action_id < self.num_actions:
                    mask[0, action.action_id] = True
            probs_by_action = F.softmax(masked_action_logits(logits, mask), dim=-1).squeeze(0).detach().cpu()
        probs = [float(probs_by_action[a.action_id]) if a.action_id < len(probs_by_action) else 0.0 for a in legal_actions]
        return ActionDistribution(legal_actions, probs, values={"value": 0.0}, metadata={"source": "model"}).normalized()
