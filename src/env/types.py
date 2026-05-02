from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.data.arc_schema import action_name


@dataclass(frozen=True)
class ARCAction:
    action_id: int
    x: int = 0
    y: int = 0
    metadata: Optional[dict[str, Any]] = None

    @property
    def name(self) -> str:
        return action_name(self.action_id)

    def key(self) -> tuple[int, int, int]:
        return (int(self.action_id), int(self.x), int(self.y))


@dataclass
class ARCObs:
    game_id: str
    game_family: str
    grid: np.ndarray
    available_actions: list[ARCAction]
    state: str
    terminal: bool
    success: bool
    score: float
    raw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ARCStepResult:
    obs: ARCObs
    action: ARCAction
    reward: float = 0.0
    valid_action: bool = True
    invalid_reason: Optional[str] = None
    exception: Optional[str] = None
