import hashlib
import json
from pathlib import Path
from typing import Iterable

from src.data.episode_reader import read_episode
from src.data.episode_schema import action_sequence_hash, final_grid_hash


def grid_hash(grid) -> str:
    return hashlib.blake2b(json.dumps(grid, sort_keys=True).encode("utf-8"), digest_size=16).hexdigest()


def episode_hash(path: str | Path) -> tuple[str, str]:
    rows = read_episode(path)
    return action_sequence_hash(rows), final_grid_hash(rows)


class DedupeStore:
    def __init__(self):
        self._seen = set()

    def add(self, game_id: str, action_hash: str, grid_hash_value: str) -> bool:
        key = (game_id, action_hash, grid_hash_value)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def contains(self, game_id: str, action_hash: str, grid_hash_value: str) -> bool:
        return (game_id, action_hash, grid_hash_value) in self._seen
