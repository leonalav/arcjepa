from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.env.types import ARCAction


@dataclass
class ActionDistribution:
    actions: list[ARCAction]
    probs: list[float]
    values: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "ActionDistribution":
        total = sum(max(0.0, p) for p in self.probs)
        if total <= 0:
            n = max(1, len(self.actions))
            return ActionDistribution(self.actions, [1.0 / n] * len(self.actions), self.values, self.metadata)
        return ActionDistribution(self.actions, [max(0.0, p) / total for p in self.probs], self.values, self.metadata)


class Agent(ABC):
    @abstractmethod
    def propose(self, obs, legal_actions: list[ARCAction], budget=None) -> ActionDistribution:
        raise NotImplementedError
