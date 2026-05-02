import json
from pathlib import Path
from typing import Iterator, Optional

from src.data.episode_schema import row_action
from src.env.types import ARCAction


def iter_episode(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_episode(path: str | Path) -> list[dict]:
    return list(iter_episode(path))


def iter_episodes(root: str | Path, outcomes: Optional[set[str] | list[str]] = None) -> Iterator[Path]:
    root = Path(root)
    allowed = set(outcomes) if outcomes is not None else None
    for path in root.rglob("*.jsonl"):
        if allowed is None or any(part in allowed for part in path.parts):
            yield path


def load_action_sequence(path: str | Path) -> list[ARCAction]:
    return [row_action(row) for row in iter_episode(path)]


def verify_episode(adapter_factory, path: str | Path) -> bool:
    actions = load_action_sequence(path)
    rows = read_episode(path)
    if not rows:
        return False
    adapter = adapter_factory(rows[0]["game_id"])
    obs, results = adapter.replay(actions)
    return bool(results and obs.success and all(r.valid_action for r in results))
