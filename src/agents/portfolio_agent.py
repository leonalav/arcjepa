from collections import defaultdict

from src.env.types import ARCAction
from .base import ActionDistribution, Agent


class PortfolioAgent(Agent):
    def __init__(self, agents: list[Agent], weights: list[float] | None = None, random_floor: float = 0.01):
        self.agents = agents
        self.weights = weights or [1.0] * len(agents)
        self.random_floor = random_floor

    def propose(self, obs, legal_actions: list[ARCAction], budget=None) -> ActionDistribution:
        scores = defaultdict(float)
        action_by_key = {a.key(): a for a in legal_actions}
        for action in legal_actions:
            scores[action.key()] += self.random_floor
        for agent, weight in zip(self.agents, self.weights):
            dist = agent.propose(obs, legal_actions, budget).normalized()
            for action, prob in zip(dist.actions, dist.probs):
                if action.key() in action_by_key:
                    scores[action.key()] += weight * prob
        actions = [action_by_key[key] for key in scores if key in action_by_key]
        probs = [scores[action.key()] for action in actions]
        return ActionDistribution(actions, probs, metadata={"source": "portfolio"}).normalized()
