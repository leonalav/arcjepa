"""
PortfolioSolver: anytime sequential portfolio of search strategies.

Pass order (cheapest → most expensive):
  1. HeuristicSolver(exploration_rate=0.05) — structural exploit
  2. HeuristicSolver(exploration_rate=0.40) — balanced explore
  3. RandomLegalSolver                       — stochastic escape valve
  4. EnvUCTSolver(num_simulations=200)       — tree search for hard cases

Returns immediately on the first solver that finds a win.
Falls back to the result with the highest final_score if no win is found.
"""

from pathlib import Path
from typing import Optional

from .heuristic import HeuristicSolver
from .random_legal import RandomLegalSolver


class PortfolioSolver:
    def __init__(
        self,
        max_steps: int = 100,
        seed: int = 0,
        uct_simulations: int = 200,
    ):
        self.max_steps = max_steps
        self.seed = seed
        self._uct_simulations = uct_simulations
        self._solvers = None

    def _build_solvers(self):
        """Deferred build so the UCT import doesn't break module-level loading."""
        from src.search.env_mcts import RealEnvMCTS
        return [
            HeuristicSolver(
                max_steps=self.max_steps,
                exploration_rate=0.05,
                seed=self.seed,
                policy_version="heuristic_exploit_v1",
            ),
            HeuristicSolver(
                max_steps=self.max_steps,
                exploration_rate=0.40,
                seed=self.seed + 1,
                policy_version="heuristic_explore_v1",
            ),
            RandomLegalSolver(
                max_steps=self.max_steps,
                seed=self.seed + 2,
            ),
            RealEnvMCTS(
                max_steps=self.max_steps,
                num_simulations=self._uct_simulations,
                seed=self.seed + 3,
            ),
        ]

    def solve(self, adapter, out_dir: str | Path, episode_id: Optional[str] = None):
        if self._solvers is None:
            self._solvers = self._build_solvers()

        solver_tags = ["exploit", "explore", "random", "uct"]
        best = None
        for idx, (solver, tag) in enumerate(zip(self._solvers, solver_tags)):
            ep_id = f"{episode_id or 'episode'}-p{idx}-{tag}"
            result = solver.solve(adapter, out_dir, episode_id=ep_id)
            if result.success:
                return result
            if best is None or result.final_score > best.final_score:
                best = result
        return best
