"""
Utility functions for MCTS inference.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


def grid_accuracy(pred_grid: torch.Tensor, target_grid: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Calculate pixel-wise accuracy between predicted and target grids.

    Args:
        pred_grid: Predicted grid [H, W] or [B, H, W]
        target_grid: Target grid [H, W] or [B, H, W]
        mask: Optional mask [H, W] or [B, H, W] to ignore certain pixels

    Returns:
        Accuracy as float in [0.0, 1.0]
    """
    if pred_grid.shape != target_grid.shape:
        raise ValueError(f"Shape mismatch: pred {pred_grid.shape} vs target {target_grid.shape}")

    matches = (pred_grid == target_grid).float()

    if mask is not None:
        matches = matches * mask
        total = mask.sum().item()
    else:
        total = matches.numel()

    if total == 0:
        return 0.0

    return (matches.sum().item() / total)


def weighted_grid_accuracy(
    pred_grid: torch.Tensor,
    target_grid: torch.Tensor,
    background_value: int = 0,
    background_weight: float = 0.5
) -> float:
    """
    Calculate weighted accuracy that penalizes errors on non-background pixels more.

    Args:
        pred_grid: Predicted grid [H, W]
        target_grid: Target grid [H, W]
        background_value: Value representing background (default 0)
        background_weight: Weight for background pixels (default 0.5)

    Returns:
        Weighted accuracy in [0.0, 1.0]
    """
    is_background = (target_grid == background_value).float()
    is_foreground = 1.0 - is_background

    matches = (pred_grid == target_grid).float()

    # Weight background matches less
    weighted_matches = matches * (is_background * background_weight + is_foreground)
    total_weight = (is_background * background_weight + is_foreground).sum().item()

    if total_weight == 0:
        return 0.0

    return (weighted_matches.sum().item() / total_weight)


def decode_action_sequence(node) -> List[Tuple[int, int, int]]:
    """
    Extract action sequence from MCTS node path (leaf to root).

    Args:
        node: MCTSNode (typically a leaf node)

    Returns:
        List of (action_type, x, y) tuples from root to leaf
    """
    path = []
    current = node

    while current.parent is not None:
        if current.action_taken is not None:
            path.append((
                current.action_taken,
                current.action_coords[0] if current.action_coords else 0,
                current.action_coords[1] if current.action_coords else 0
            ))
        current = current.parent

    # Reverse to get root-to-leaf order
    return list(reversed(path))


def visualize_search_tree(root_node, max_depth: int = 5, top_k: int = 3) -> str:
    """
    Generate a text visualization of the MCTS search tree.

    Args:
        root_node: Root MCTSNode
        max_depth: Maximum depth to visualize
        top_k: Show only top K children by visit count

    Returns:
        String representation of the tree
    """
    lines = []

    def _traverse(node, depth: int, prefix: str = ""):
        if depth > max_depth:
            return

        # Node info
        q_val = node.q_value() if node.visits > 0 else 0.0
        action_str = f"a{node.action_taken}" if node.action_taken is not None else "ROOT"

        line = f"{prefix}{action_str} | N={node.visits} Q={q_val:.3f}"
        if node.is_terminal:
            line += " [WIN]"
        lines.append(line)

        # Sort children by visit count
        if node.children:
            sorted_children = sorted(
                node.children.items(),
                key=lambda x: x[1].visits,
                reverse=True
            )[:top_k]

            for i, ((action, coords), child) in enumerate(sorted_children):
                is_last = (i == len(sorted_children) - 1)
                child_prefix = prefix + ("└── " if is_last else "├── ")
                next_prefix = prefix + ("    " if is_last else "│   ")

                _traverse(child, depth + 1, child_prefix)

    _traverse(root_node, 0)
    return "\n".join(lines)


def compute_rollout_statistics(root_node) -> dict:
    """
    Compute statistics about the MCTS search tree.

    Args:
        root_node: Root MCTSNode

    Returns:
        Dictionary with tree statistics
    """
    total_nodes = 0
    max_depth = 0
    terminal_nodes = 0
    total_value = 0.0

    def _traverse(node, depth):
        nonlocal total_nodes, max_depth, terminal_nodes, total_value

        total_nodes += 1
        max_depth = max(max_depth, depth)

        if node.is_terminal:
            terminal_nodes += 1

        total_value += node.total_value

        for child in node.children.values():
            _traverse(child, depth + 1)

    _traverse(root_node, 0)

    return {
        "total_nodes": total_nodes,
        "max_depth": max_depth,
        "terminal_nodes": terminal_nodes,
        "avg_value": total_value / total_nodes if total_nodes > 0 else 0.0,
        "root_visits": root_node.visits,
        "root_q": root_node.q_value() if root_node.visits > 0 else 0.0,
    }


def format_action_sequence(actions: List[Tuple[int, int, int]]) -> str:
    """
    Format action sequence for human-readable display.

    Args:
        actions: List of (action_type, x, y) tuples

    Returns:
        Formatted string
    """
    action_names = {
        0: "NONE",
        1: "ACTION1",
        2: "ACTION2",
        3: "ACTION3",
        4: "ACTION4",
        5: "ACTION5",
        6: "ACTION6",
        7: "ACTION7",
        8: "SUBMIT",
    }

    lines = []
    for i, (action, x, y) in enumerate(actions):
        name = action_names.get(action, f"UNKNOWN({action})")
        lines.append(f"Step {i+1}: {name} at ({x}, {y})")

    return "\n".join(lines)
