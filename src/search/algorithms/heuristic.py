"""
HeuristicSolver: wraps ARCHeuristicPolicy with the standard solve() interface.

Replaces the blind RandomLegalSolver with pattern-informed action selection
(symmetry detection, flood-fill targeting, object copying) while keeping a
small exploration_rate for stochastic escape from local loops.
"""

from pathlib import Path
from typing import Optional

from src.data.arc_schema import action_uses_coordinates
from src.data.episode_writer import EpisodeWriter, move_episode_to_outcome
from src.data.heuristic_policy import ARCHeuristicPolicy
from src.env.types import ARCAction
from src.search.results import EpisodeResult


class HeuristicSolver:
    """
    Policy-driven solver using ARCHeuristicPolicy.

    Chooses actions via:
      - Symmetry detection  → ACTION1
      - Repetition / copy   → ACTION2
      - Flood fill (objects)→ ACTION3
      - Frontier painting   → ACTION6 with smart coords
      - Exploration fallback → random legal action

    Args:
        max_steps: Episode step budget.
        exploration_rate: Probability of choosing a random legal action
            instead of the heuristic pick (default 0.15 — enough to escape
            loops without diluting data quality).
        seed: RNG seed for reproducibility.
        policy_version: Written into every JSONL row for lineage tracking.
    """

    def __init__(
        self,
        max_steps: int = 100,
        exploration_rate: float = 0.15,
        seed: int = 0,
        policy_version: str = "heuristic_v1",
    ):
        self.max_steps = max_steps
        self.exploration_rate = exploration_rate
        self.seed = seed
        self.policy_version = policy_version
        self.policy = ARCHeuristicPolicy(
            exploration_rate=exploration_rate,
        )
        import random
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public interface (mirrors RandomLegalSolver.solve exactly)
    # ------------------------------------------------------------------

    def solve(
        self,
        adapter,
        out_dir: str | Path,
        episode_id: Optional[str] = None,
    ) -> EpisodeResult:
        out_dir = Path(out_dir)
        obs = adapter.reset()

        episode_id = episode_id or (
            f"{obs.game_id}-heuristic-{self._rng.randrange(10**12):012d}"
        )

        temp_dir = out_dir / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{episode_id}.jsonl"

        valid_actions = 0
        invalid_actions = 0
        action_sequence = []
        terminal = False
        success = False
        final_score = obs.score

        with EpisodeWriter(
            temp_path,
            episode_id=episode_id,
            policy_version=self.policy_version,
            search_algorithm="heuristic",
            search_budget={"max_steps": self.max_steps},
        ) as writer:
            for step in range(self.max_steps):
                if not obs.available_actions:
                    terminal = obs.terminal
                    success = obs.success
                    final_score = obs.score
                    break

                action = self._choose_action(obs)
                before = obs
                result = adapter.step(action)
                writer.write_transition(step=step, before=before, result=result)
                action_sequence.append(action)

                if result.valid_action:
                    valid_actions += 1
                else:
                    invalid_actions += 1

                obs = result.obs
                terminal = obs.terminal
                success = obs.success
                final_score = obs.score

                if obs.terminal or not result.valid_action:
                    break

        outcome = "wins" if success else "failed" if terminal else "partial"
        final_path = move_episode_to_outcome(
            temp_path, out_dir, outcome, obs.game_id, episode_id
        )
        total = valid_actions + invalid_actions
        return EpisodeResult(
            episode_id=episode_id,
            game_id=obs.game_id,
            success=success,
            terminal=terminal,
            steps=len(action_sequence),
            valid_actions=valid_actions,
            invalid_actions=invalid_actions,
            valid_action_rate=valid_actions / max(1, total),
            path=final_path,
            action_sequence=action_sequence,
            final_score=final_score,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _choose_action(self, obs) -> ARCAction:
        """Delegate to ARCHeuristicPolicy, then convert to ARCAction."""
        available_names = [a.name for a in obs.available_actions]
        action_enum, x, y = self.policy.select_action(
            obs.grid,
            step=int(obs.metadata.get("step", 0)) if getattr(obs, "metadata", None) else 0,
            available_actions=available_names,
            game_id=obs.game_id,
        )
        action_idx = self.policy._action_idx(action_enum)

        # Validate against available legal actions
        legal_ids = {a.action_id for a in obs.available_actions}
        if action_idx not in legal_ids:
            # Heuristic proposed an illegal action — fall back to random legal
            candidates = [a for a in obs.available_actions if a.action_id > 0]
            if not candidates:
                candidates = obs.available_actions
            fallback = self._rng.choice(candidates)
            if action_uses_coordinates(fallback.action_id):
                import numpy as np
                active = (obs.grid != 0).nonzero()
                if len(active[0]) > 0:
                    idx = self._rng.randrange(len(active[0]))
                    return ARCAction(fallback.action_id, int(active[1][idx]), int(active[0][idx]))
                return ARCAction(fallback.action_id, self._rng.randrange(64), self._rng.randrange(64))
            return ARCAction(fallback.action_id)

        if not action_uses_coordinates(action_idx):
            x, y = 0, 0
        return ARCAction(action_idx, int(x), int(y))
