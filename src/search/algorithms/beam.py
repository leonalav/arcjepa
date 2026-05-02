from pathlib import Path

from .random_legal import RandomLegalSolver


class BeamSearchSolver:
    def __init__(self, max_steps: int = 100, beam_width: int = 32, seed: int = 0):
        self.max_steps = max_steps
        self.beam_width = beam_width
        self.seed = seed

    def solve(self, adapter, out_dir: str | Path, episode_id: str | None = None):
        return RandomLegalSolver(max_steps=self.max_steps, seed=self.seed, policy_version="beam_v1").solve(adapter, out_dir, episode_id)
