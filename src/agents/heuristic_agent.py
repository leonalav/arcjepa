from src.data.heuristic_policy import ARCHeuristicPolicy
from src.env.types import ARCAction
from .base import ActionDistribution, Agent


class HeuristicAgent(Agent):
    def __init__(self, exploration_rate: float = 0.0):
        self.policy = ARCHeuristicPolicy(exploration_rate=exploration_rate)

    def propose(self, obs, legal_actions: list[ARCAction], budget=None) -> ActionDistribution:
        if not legal_actions:
            return ActionDistribution([], [])
        action_enum, x, y = self.policy.select_action(
            obs.grid,
            step=int(obs.metadata.get("step", 0)) if getattr(obs, "metadata", None) else 0,
            available_actions=[a.name for a in legal_actions],
            game_id=obs.game_id,
        )
        chosen = ARCAction(self.policy._action_idx(action_enum), x, y)
        ordered = [chosen] + [a for a in legal_actions if a.action_id != chosen.action_id]
        probs = [1.0] + [0.1] * (len(ordered) - 1)
        return ActionDistribution(ordered, probs, metadata={"source": "heuristic"}).normalized()
