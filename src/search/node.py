from dataclasses import dataclass, field
from math import sqrt
from typing import Optional

from src.env.types import ARCAction


@dataclass
class EnvMCTSNode:
    prefix: tuple[ARCAction, ...]
    parent: Optional["EnvMCTSNode"] = None
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    terminal: bool = False
    success: bool = False
    children: dict[tuple[int, int, int], "EnvMCTSNode"] = field(default_factory=dict)

    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def puct_score(self, c_puct: float = 1.4) -> float:
        if self.parent is None:
            return self.q_value()
        return self.q_value() + c_puct * self.prior * sqrt(max(1, self.parent.visits)) / (1 + self.visits)

    def select_child(self, c_puct: float = 1.4) -> "EnvMCTSNode":
        return max(self.children.values(), key=lambda child: child.puct_score(c_puct))

    def add_child(self, action: ARCAction, prior: float = 1.0) -> "EnvMCTSNode":
        child = EnvMCTSNode(prefix=self.prefix + (action,), parent=self, prior=prior)
        self.children[action.key()] = child
        return child

    def backpropagate(self, value: float):
        self.visits += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.backpropagate(value)
