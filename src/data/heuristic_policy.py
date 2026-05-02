"""
Heuristic policy for generating ARC-AGI-3 trajectories.

The policy proposes pattern-based moves, then filters every proposal through the
frame's available action set so generated recordings do not train on illegal moves.
"""

import numpy as np
from typing import Any, Iterable, Optional, Tuple
from arcengine import GameAction

from .arc_schema import (
    DEFAULT_NUM_ACTIONS,
    action_name,
    action_uses_coordinates,
    available_indices,
    parse_action_name,
)


class ARCHeuristicPolicy:
    def __init__(self, exploration_rate: float = 0.2, num_actions: int = DEFAULT_NUM_ACTIONS):
        self.exploration_rate = exploration_rate
        self.num_actions = num_actions
        self.action_history = []

    def select_action(
        self,
        grid: np.ndarray,
        step: int = 0,
        available_actions: Optional[Iterable[Any]] = None,
        game_id: Optional[str] = None,
        objective: Optional[Any] = None,
    ) -> Tuple[GameAction, int, int]:
        if np.random.random() < self.exploration_rate:
            return self._random_action(grid, available_actions)

        fg_ratio = np.mean(grid != 0)
        candidates = []

        if fg_ratio < 0.1:
            candidates.append(self._paint_action(grid))
        elif self._has_symmetry(grid):
            candidates.append(self._mirror_action(grid))
        elif self._has_repetition(grid):
            candidates.append(self._copy_action(grid))
        elif self._has_isolated_objects(grid):
            candidates.append(self._flood_fill_action(grid))
        elif fg_ratio < 0.5:
            candidates.append(self._paint_action(grid))
        else:
            candidates.append(self._transform_action(grid))

        candidates.extend([
            self._paint_action(grid),
            self._transform_action(grid),
            self._mirror_action(grid),
            self._copy_action(grid),
        ])
        return self._choose_valid(candidates, grid, available_actions)

    def _to_game_action(self, action_idx: int):
        name = action_name(action_idx)
        if hasattr(GameAction, name):
            return getattr(GameAction, name)
        return GameAction.ACTION1

    def _action_idx(self, action: Any) -> int:
        return parse_action_name(action, num_actions=self.num_actions, strict=False)

    def _valid_set(self, available_actions: Optional[Iterable[Any]]) -> set[int]:
        return set(available_indices(available_actions, num_actions=self.num_actions))

    def _choose_valid(self, candidates, grid: np.ndarray, available_actions: Optional[Iterable[Any]]):
        valid = self._valid_set(available_actions)
        for action, x, y in candidates:
            idx = self._action_idx(action)
            if idx in valid:
                if not action_uses_coordinates(idx):
                    x, y = 0, 0
                return action, int(x), int(y)
        return self._random_action(grid, available_actions)

    def _has_symmetry(self, grid: np.ndarray) -> bool:
        if np.array_equal(grid, np.flip(grid, axis=0)):
            return True
        if np.array_equal(grid, np.flip(grid, axis=1)):
            return True
        return False

    def _has_repetition(self, grid: np.ndarray) -> bool:
        h, w = grid.shape
        if h < 4 or w < 4:
            return False
        block_h, block_w = 2, 2
        if h % block_h != 0 or w % block_w != 0:
            return False
        first_block = grid[:block_h, :block_w]
        for i in range(0, h, block_h):
            for j in range(0, w, block_w):
                if not np.array_equal(grid[i:i+block_h, j:j+block_w], first_block):
                    return False
        return True

    def _has_isolated_objects(self, grid: np.ndarray) -> bool:
        try:
            from scipy.ndimage import label
            foreground = (grid != 0)
            _, num_features = label(foreground)
            return num_features >= 2
        except ImportError:
            return False

    def _mirror_action(self, grid: np.ndarray) -> Tuple[GameAction, int, int]:
        return (GameAction.ACTION1, 0, 0)

    def _copy_action(self, grid: np.ndarray) -> Tuple[GameAction, int, int]:
        return (GameAction.ACTION2, 0, 0)

    def _flood_fill_action(self, grid: np.ndarray) -> Tuple[GameAction, int, int]:
        try:
            from scipy.ndimage import label
            foreground = (grid != 0)
            labeled, num = label(foreground)
            if num > 0:
                sizes = [(labeled == i).sum() for i in range(1, num+1)]
                largest = np.argmax(sizes) + 1
                coords = np.argwhere(labeled == largest)
                center = coords.mean(axis=0).astype(int)
                return (GameAction.ACTION3, int(center[1]), int(center[0]))
        except ImportError:
            pass
        return (GameAction.ACTION3, 0, 0)

    def _paint_action(self, grid: np.ndarray) -> Tuple[GameAction, int, int]:
        foreground = (grid != 0)
        empty = (grid == 0)
        if foreground.any():
            try:
                from scipy.ndimage import binary_dilation
                candidates = binary_dilation(foreground) & empty
                if candidates.any():
                    coords = np.argwhere(candidates)
                    y, x = coords[np.random.randint(len(coords))]
                    return (GameAction.ACTION6, int(x), int(y))
            except ImportError:
                pass
        empty_coords = np.argwhere(empty)
        if len(empty_coords) > 0:
            y, x = empty_coords[np.random.randint(len(empty_coords))]
            return (GameAction.ACTION6, int(x), int(y))
        return (GameAction.ACTION6, 0, 0)

    def _transform_action(self, grid: np.ndarray) -> Tuple[GameAction, int, int]:
        return (GameAction.ACTION4, 0, 0)

    def _random_action(self, grid: np.ndarray, available_actions: Optional[Iterable[Any]] = None) -> Tuple[GameAction, int, int]:
        valid = [idx for idx in self._valid_set(available_actions) if idx > 0]
        if not valid:
            valid = [1]
        action_idx = int(np.random.choice(valid))
        action = self._to_game_action(action_idx)
        if action_uses_coordinates(action_idx):
            h, w = grid.shape
            x = np.random.randint(0, max(1, min(w, 64)))
            y = np.random.randint(0, max(1, min(h, 64)))
            return (action, int(x), int(y))
        return (action, 0, 0)
