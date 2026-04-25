"""
Heuristic policy for generating meaningful ARC trajectories.

This policy detects patterns in grid states and selects appropriate actions
to create trajectories with causal structure, avoiding random meaningless actions.
"""

import numpy as np
from typing import Tuple
from arcengine import GameAction


class ARCHeuristicPolicy:
    """
    Heuristic policy for ARC puzzle solving.

    Detects patterns (symmetry, repetition, isolated objects) and applies
    appropriate transformations to generate meaningful state transitions.
    """

    def __init__(self, exploration_rate: float = 0.2):
        """
        Args:
            exploration_rate: Probability of taking random exploratory action
        """
        self.exploration_rate = exploration_rate
        self.action_history = []

    def select_action(
        self,
        grid: np.ndarray,
        step: int = 0
    ) -> Tuple[GameAction, int, int]:
        """
        Select action based on grid patterns.

        Args:
            grid: Current grid state [H, W]
            step: Current step number

        Returns:
            (action_type, x_coord, y_coord)
        """
        # Random exploration
        if np.random.random() < self.exploration_rate:
            return self._random_action(grid)

        # Pattern-based action selection
        fg_ratio = np.mean(grid != 0)

        if fg_ratio < 0.1:
            # Mostly empty - add patterns
            return self._paint_action(grid)
        elif self._has_symmetry(grid):
            # Has symmetry - apply mirror
            return self._mirror_action(grid)
        elif self._has_repetition(grid):
            # Has repetition - copy pattern
            return self._copy_action(grid)
        elif self._has_isolated_objects(grid):
            # Has isolated objects - flood fill
            return self._flood_fill_action(grid)
        elif fg_ratio < 0.5:
            # Sparse foreground - add more
            return self._paint_action(grid)
        else:
            # Dense - apply transformation
            return self._transform_action(grid)

    def _has_symmetry(self, grid: np.ndarray) -> bool:
        """Detect horizontal or vertical symmetry."""
        # Check horizontal symmetry
        if np.array_equal(grid, np.flip(grid, axis=0)):
            return True
        # Check vertical symmetry
        if np.array_equal(grid, np.flip(grid, axis=1)):
            return True
        return False

    def _has_repetition(self, grid: np.ndarray) -> bool:
        """Detect repeating patterns."""
        h, w = grid.shape
        if h < 4 or w < 4:
            return False

        # Check for 2x2 repeating blocks
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
        """Detect isolated foreground objects."""
        try:
            from scipy.ndimage import label
            foreground = (grid != 0)
            labeled, num_features = label(foreground)
            return num_features >= 2
        except ImportError:
            # Fallback if scipy not available
            return False

    def _mirror_action(self, grid: np.ndarray) -> Tuple[GameAction, int, int]:
        """Apply mirror transformation."""
        return (GameAction.ACTION1, 0, 0)

    def _copy_action(self, grid: np.ndarray) -> Tuple[GameAction, int, int]:
        """Copy pattern action."""
        return (GameAction.ACTION2, 0, 0)

    def _flood_fill_action(self, grid: np.ndarray) -> Tuple[GameAction, int, int]:
        """Flood fill isolated regions."""
        try:
            from scipy.ndimage import label
            foreground = (grid != 0)
            labeled, num = label(foreground)

            if num > 0:
                # Find center of largest component
                sizes = [(labeled == i).sum() for i in range(1, num+1)]
                largest = np.argmax(sizes) + 1
                coords = np.argwhere(labeled == largest)
                center = coords.mean(axis=0).astype(int)
                return (GameAction.ACTION3, int(center[1]), int(center[0]))
        except ImportError:
            pass

        return (GameAction.ACTION3, 0, 0)

    def _paint_action(self, grid: np.ndarray) -> Tuple[GameAction, int, int]:
        """Paint action at strategic location."""
        foreground = (grid != 0)
        empty = (grid == 0)

        if foreground.any():
            try:
                from scipy.ndimage import binary_dilation
                # Find empty cells adjacent to foreground
                dilated = binary_dilation(foreground)
                candidates = dilated & empty

                if candidates.any():
                    coords = np.argwhere(candidates)
                    idx = np.random.randint(len(coords))
                    y, x = coords[idx]
                    return (GameAction.ACTION6, int(x), int(y))
            except ImportError:
                pass

        # Random empty cell
        empty_coords = np.argwhere(empty)
        if len(empty_coords) > 0:
            idx = np.random.randint(len(empty_coords))
            y, x = empty_coords[idx]
            return (GameAction.ACTION6, int(x), int(y))

        return (GameAction.ACTION6, 0, 0)

    def _transform_action(self, grid: np.ndarray) -> Tuple[GameAction, int, int]:
        """Apply general transformation."""
        return (GameAction.ACTION4, 0, 0)

    def _random_action(self, grid: np.ndarray) -> Tuple[GameAction, int, int]:
        """Random exploration action."""
        action = np.random.choice([
            GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
            GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6
        ])

        if action == GameAction.ACTION6:
            h, w = grid.shape
            x = np.random.randint(0, min(w, 64))
            y = np.random.randint(0, min(h, 64))
            return (action, x, y)

        return (action, 0, 0)
