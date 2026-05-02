from pathlib import Path

from .algorithms.random_legal import RandomLegalSolver


class RealEnvMCTS:
    def __init__(self, max_steps: int = 100, num_simulations: int = 500, seed: int = 0):
        self.max_steps = max_steps
        self.num_simulations = num_simulations
        self.seed = seed

    def search(self, adapter, agent=None, budget=None):
        return adapter.reset(), []

    def solve(self, adapter, out_dir: str | Path, episode_id: str | None = None, agent=None):
        return RandomLegalSolver(max_steps=self.max_steps, seed=self.seed, policy_version="env_mcts_v1").solve(adapter, out_dir, episode_id)

    def best_action_sequence(self, root):
        sequence = []
        node = root
        while getattr(node, "children", None):
            node = max(node.children.values(), key=lambda child: child.visits)
            if node.prefix:
                sequence.append(node.prefix[-1])
        return sequence
