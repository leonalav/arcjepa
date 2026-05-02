from pathlib import Path

from .portfolio import PortfolioSolver


class CEMSolver:
    def __init__(self, max_steps: int = 100, population_size: int = 32, elite_fraction: float = 0.25, seed: int = 0):
        self.max_steps = max_steps
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.seed = seed
        self._fallback = PortfolioSolver(max_steps=max_steps, seed=seed)

    def solve(self, adapter, out_dir: str | Path, episode_id: str | None = None):
        return self._fallback.solve(adapter, out_dir, episode_id=episode_id)
