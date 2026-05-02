import random
import tempfile
from pathlib import Path
from typing import Optional

from src.data.arc_schema import action_uses_coordinates
from src.data.episode_writer import EpisodeWriter, move_episode_to_outcome
from src.env.types import ARCAction
from src.search.results import EpisodeResult


class RandomLegalSolver:
    def __init__(self, max_steps: int = 100, seed: int = 0, policy_version: str = "random_legal_v1"):
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.policy_version = policy_version

    def solve(self, adapter, out_dir: str | Path, episode_id: Optional[str] = None) -> EpisodeResult:
        out_dir = Path(out_dir)
        obs = adapter.reset()
        episode_id = episode_id or f"{obs.game_id}-random-{self.rng.randrange(10**12):012d}"
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
            search_algorithm="random_legal",
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
        final_path = move_episode_to_outcome(temp_path, out_dir, outcome, obs.game_id, episode_id)
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

    def _choose_action(self, obs) -> ARCAction:
        candidates = [a for a in obs.available_actions if a.action_id > 0]
        if not candidates:
            candidates = obs.available_actions
        base = self.rng.choice(candidates)
        if action_uses_coordinates(base.action_id):
            active = (obs.grid != 0).nonzero()
            if len(active[0]) > 0:
                idx = self.rng.randrange(len(active[0]))
                y = int(active[0][idx])
                x = int(active[1][idx])
            else:
                x = self.rng.randrange(64)
                y = self.rng.randrange(64)
            return ARCAction(base.action_id, x=x, y=y)
        return ARCAction(base.action_id)
