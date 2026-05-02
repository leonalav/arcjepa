from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EpisodeResult:
    episode_id: str
    game_id: str
    success: bool
    terminal: bool
    steps: int
    valid_actions: int
    invalid_actions: int
    valid_action_rate: float
    path: Path
    action_sequence: list
    final_score: float = 0.0
    metadata: dict = field(default_factory=dict)
