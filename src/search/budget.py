from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SearchBudget:
    max_steps: int = 100
    max_nodes: int = 10000
    max_env_steps: int = 100000
    max_wallclock_sec: float = 3600.0
    episodes_per_game: int = 1000
    beam_width: int = 64
    num_simulations: int = 500
    progressive_widening_base: int = 8
    progressive_widening_alpha: float = 0.5
    seed: int = 0

    def expired(self, stats: dict[str, Any]) -> bool:
        return (
            stats.get("steps", 0) >= self.max_steps or
            stats.get("nodes", 0) >= self.max_nodes or
            stats.get("env_steps", 0) >= self.max_env_steps or
            stats.get("wallclock_sec", 0.0) >= self.max_wallclock_sec
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
