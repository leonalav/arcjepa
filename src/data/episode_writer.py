import json
import shutil
from pathlib import Path
from typing import Any

from src.data.episode_schema import transition_to_row
from src.env.types import ARCObs, ARCStepResult


def episode_output_path(out_dir: str | Path, outcome: str, game_id: str, episode_id: str) -> Path:
    safe_game = str(game_id).replace("/", "_").replace("\\", "_")
    return Path(out_dir) / outcome / f"game_id={safe_game}" / f"{episode_id}.jsonl"


class EpisodeWriter:
    def __init__(
        self,
        path: str | Path,
        episode_id: str,
        policy_version: str,
        search_algorithm: str,
        search_budget: dict[str, Any],
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id
        self.policy_version = policy_version
        self.search_algorithm = search_algorithm
        self.search_budget = search_budget
        self._f = self.path.open("w", encoding="utf-8", newline="\n")

    def write_transition(self, step: int, before: ARCObs, result: ARCStepResult, **extras: Any) -> dict[str, Any]:
        row = transition_to_row(
            episode_id=self.episode_id,
            policy_version=self.policy_version,
            search_algorithm=self.search_algorithm,
            search_budget=self.search_budget,
            step=step,
            before=before,
            result=result,
            **extras,
        )
        self._f.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._f.flush()
        return row

    def close(self):
        if not self._f.closed:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def move_episode_to_outcome(path: str | Path, out_dir: str | Path, outcome: str, game_id: str, episode_id: str) -> Path:
    path = Path(path)
    target = episode_output_path(out_dir, outcome, game_id, episode_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() != target.resolve():
        shutil.move(str(path), str(target))
    return target
