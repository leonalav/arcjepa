from src.search.env_mcts import RealEnvMCTS


class EnvUCTSolver:
    def __init__(self, max_steps: int = 100, num_simulations: int = 500, seed: int = 0):
        self.mcts = RealEnvMCTS(max_steps=max_steps, num_simulations=num_simulations, seed=seed)

    def solve(self, adapter, out_dir, episode_id=None, agent=None):
        return self.mcts.solve(adapter, out_dir, episode_id=episode_id, agent=agent)
