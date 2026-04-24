"""
MCTSNode: Data structure for Monte Carlo Tree Search in latent space.
"""

import torch
import math
from typing import Optional, Dict, Tuple, Any


class MCTSNode:
    """
    Node in the MCTS search tree.

    Stores latent state, GDN recurrent cache, tree structure, and visit statistics.
    """

    def __init__(
        self,
        s_t: torch.Tensor,
        rnn_state: Optional[Any] = None,
        parent: Optional['MCTSNode'] = None,
        action_taken: Optional[int] = None,
        action_coords: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize MCTS node.

        Args:
            s_t: Latent state vector [1, d_model]
            rnn_state: GDN recurrent cache (tuple or tensor)
            parent: Parent node in the tree
            action_taken: Action index (1-7) that led to this node
            action_coords: (x, y) coordinates for the action
        """
        # Latent state
        self.s_t = s_t
        self.rnn_state = rnn_state

        # Tree structure
        self.parent = parent
        self.action_taken = action_taken
        self.action_coords = action_coords
        self.children: Dict[Tuple[int, int, int], 'MCTSNode'] = {}  # (action, x, y) -> child

        # Visit statistics
        self.visits = 0
        self.total_value = 0.0

        # Terminal state flag
        self.is_terminal = False

    def q_value(self) -> float:
        """
        Calculate Q-value (average reward).

        Returns:
            Q = W / N (total_value / visits)
        """
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def uct_score(self, c_puct: float, parent_visits: int) -> float:
        """
        Calculate Upper Confidence Bound for Trees (UCT) score.

        UCT = Q + c_puct * sqrt(ln(N_parent) / N_child)

        Args:
            c_puct: Exploration constant
            parent_visits: Number of visits to parent node

        Returns:
            UCT score (higher = more promising)
        """
        if self.visits == 0:
            # Unvisited nodes get infinite exploration bonus
            return float('inf')

        exploitation = self.q_value()
        exploration = c_puct * math.sqrt(math.log(parent_visits) / self.visits)

        return exploitation + exploration

    def is_fully_expanded(self, valid_action_coords: list) -> bool:
        """
        Check if all valid actions have been tried from this node.

        Args:
            valid_action_coords: List of (action, x, y) tuples to consider

        Returns:
            True if all actions have been expanded
        """
        return len(self.children) >= len(valid_action_coords)

    def select_best_child(self, c_puct: float) -> 'MCTSNode':
        """
        Select child with highest UCT score.

        Args:
            c_puct: Exploration constant

        Returns:
            Child node with highest UCT
        """
        if not self.children:
            raise ValueError("Cannot select from node with no children")

        best_child = None
        best_score = -float('inf')

        for child in self.children.values():
            score = child.uct_score(c_puct, self.visits)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def get_untried_actions(self, valid_action_coords: list) -> list:
        """
        Get list of actions that haven't been tried yet.

        Args:
            valid_action_coords: List of (action, x, y) tuples to consider

        Returns:
            List of untried (action, x, y) tuples
        """
        tried = set(self.children.keys())
        untried = [action for action in valid_action_coords if action not in tried]
        return untried

    def add_child(
        self,
        action: int,
        coords: Tuple[int, int],
        s_next: torch.Tensor,
        rnn_state_next: Optional[Any]
    ) -> 'MCTSNode':
        """
        Add a child node for the given action.

        Args:
            action: Action index (1-7)
            coords: (x, y) coordinates
            s_next: Next latent state [1, d_model]
            rnn_state_next: Next GDN recurrent cache

        Returns:
            The newly created child node
        """
        key = (action, coords[0], coords[1])

        if key in self.children:
            raise ValueError(f"Child for action {key} already exists")

        child = MCTSNode(
            s_t=s_next,
            rnn_state=rnn_state_next,
            parent=self,
            action_taken=action,
            action_coords=coords
        )

        self.children[key] = child
        return child

    def backpropagate(self, reward: float):
        """
        Backpropagate reward up the tree.

        Args:
            reward: Reward value to propagate (typically in [0.0, 1.0])
        """
        node = self
        while node is not None:
            node.visits += 1
            node.total_value += reward
            node = node.parent

    def get_best_action(self) -> Optional[Tuple[int, int, int]]:
        """
        Get the best action from this node based on visit counts.

        Returns:
            (action, x, y) tuple with highest visit count, or None if no children
        """
        if not self.children:
            return None

        best_child_key = max(
            self.children.keys(),
            key=lambda k: self.children[k].visits
        )

        return best_child_key

    def get_path_to_root(self) -> list:
        """
        Get the path from this node to the root.

        Returns:
            List of nodes from root to this node
        """
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def __repr__(self) -> str:
        action_str = f"a{self.action_taken}" if self.action_taken is not None else "ROOT"
        return f"MCTSNode({action_str}, N={self.visits}, Q={self.q_value():.3f})"
