from pathlib import Path

from .beam import BeamSearchSolver
from .random_legal import RandomLegalSolver


class PortfolioSolver:
    def __init__(self, max_steps: int = 100, seed: int = 0):
        self.solvers = [
            RandomLegalSolver(max_steps=max_steps, seed=seed),
            BeamSearchSolver(max_steps=max_steps, seed=seed + 1),
        ]

    def solve(self, adapter, out_dir: str | Path, episode_id: str | None = None):
        best = None
        for idx, solver in enumerate(self.solvers):
            result = solver.solve(adapter, out_dir, episode_id=f"{episode_id or 'episode'}-p{idx}")
            if result.success:
                return result
            if best is None or result.final_score > best.final_score:
                best = result
        return best
