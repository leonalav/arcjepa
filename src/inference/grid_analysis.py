"""
Grid analysis utilities for unsupervised evaluation.
Provides heuristics for assessing grid quality without ground truth.
"""

import torch
import numpy as np
from typing import Tuple, List, Set


def detect_objects(grid: torch.Tensor) -> List[Set[Tuple[int, int]]]:
    """
    Detect connected components (objects) in the grid.

    Args:
        grid: Grid tensor [H, W]

    Returns:
        List of sets, each containing (x, y) coordinates of one object
    """
    grid_np = grid.cpu().numpy()
    h, w = grid_np.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []

    def flood_fill(start_x, start_y, color):
        """BFS flood fill to find connected component."""
        if color == 0:  # Skip background
            return set()

        stack = [(start_x, start_y)]
        component = set()

        while stack:
            x, y = stack.pop()
            if x < 0 or x >= h or y < 0 or y >= w:
                continue
            if visited[x, y] or grid_np[x, y] != color:
                continue

            visited[x, y] = True
            component.add((x, y))

            # 4-connectivity
            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])

        return component

    for x in range(h):
        for y in range(w):
            if not visited[x, y] and grid_np[x, y] != 0:
                obj = flood_fill(x, y, grid_np[x, y])
                if obj:
                    objects.append(obj)

    return objects


def check_symmetry(grid: torch.Tensor) -> float:
    """
    Check for horizontal, vertical, and diagonal symmetry.

    Args:
        grid: Grid tensor [H, W]

    Returns:
        Symmetry score in [0.0, 1.0]
    """
    grid_np = grid.cpu().numpy()
    h, w = grid_np.shape

    # Horizontal symmetry
    h_sym = np.mean(grid_np == np.flip(grid_np, axis=0))

    # Vertical symmetry
    v_sym = np.mean(grid_np == np.flip(grid_np, axis=1))

    # Diagonal symmetry (if square)
    if h == w:
        d_sym = np.mean(grid_np == grid_np.T)
    else:
        d_sym = 0.0

    # Return max symmetry found
    return float(max(h_sym, v_sym, d_sym))


def check_completion(grid: torch.Tensor) -> float:
    """
    Heuristic for whether grid looks "complete".

    Checks:
    - Low entropy (few colors used consistently)
    - No isolated single pixels (noise)
    - Balanced color distribution

    Args:
        grid: Grid tensor [H, W]

    Returns:
        Completion score in [0.0, 1.0]
    """
    grid_np = grid.cpu().numpy()

    # 1. Color entropy (prefer fewer distinct colors)
    unique_colors = len(np.unique(grid_np))
    color_score = 1.0 - min(unique_colors / 16.0, 1.0)

    # 2. Isolated pixel penalty
    isolated_count = 0
    h, w = grid_np.shape
    for x in range(1, h-1):
        for y in range(1, w-1):
            if grid_np[x, y] != 0:  # Non-background
                neighbors = [
                    grid_np[x-1, y], grid_np[x+1, y],
                    grid_np[x, y-1], grid_np[x, y+1]
                ]
                if all(n != grid_np[x, y] for n in neighbors):
                    isolated_count += 1

    isolation_score = 1.0 - min(isolated_count / (h * w * 0.1), 1.0)

    # 3. Non-background ratio (prefer some content)
    non_bg = np.sum(grid_np != 0) / (h * w)
    content_score = min(non_bg * 2, 1.0)  # Penalize too empty

    return 0.4 * color_score + 0.3 * isolation_score + 0.3 * content_score


def check_consistency(pred_grid: torch.Tensor, input_grid: torch.Tensor) -> float:
    """
    Check if prediction is consistent with input structure.

    Args:
        pred_grid: Predicted grid [H, W]
        input_grid: Input grid [H, W]

    Returns:
        Consistency score in [0.0, 1.0]
    """
    pred_np = pred_grid.cpu().numpy()
    input_np = input_grid.cpu().numpy()

    # 1. Preserved input pixels (if input had content, did we keep it?)
    input_mask = input_np != 0
    if input_mask.sum() > 0:
        preserved = np.sum((pred_np == input_np) & input_mask) / input_mask.sum()
    else:
        preserved = 1.0

    # 2. Similar color palette
    input_colors = set(input_np.flatten())
    pred_colors = set(pred_np.flatten())
    color_overlap = len(input_colors & pred_colors) / max(len(input_colors | pred_colors), 1)

    # 3. Similar object count
    input_objs = len(detect_objects(input_grid))
    pred_objs = len(detect_objects(pred_grid))
    obj_ratio = min(input_objs, pred_objs) / max(input_objs, pred_objs, 1)

    return 0.4 * preserved + 0.3 * color_overlap + 0.3 * obj_ratio


def find_edges(grid: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Find edge pixels (boundaries of colored regions).

    Args:
        grid: Grid tensor [H, W]

    Returns:
        List of (x, y) edge coordinates
    """
    grid_np = grid.cpu().numpy()
    h, w = grid_np.shape
    edges = []

    for x in range(h):
        for y in range(w):
            if grid_np[x, y] != 0:  # Non-background
                # Check if adjacent to different color or background
                neighbors = []
                if x > 0: neighbors.append(grid_np[x-1, y])
                if x < h-1: neighbors.append(grid_np[x+1, y])
                if y > 0: neighbors.append(grid_np[x, y-1])
                if y < w-1: neighbors.append(grid_np[x, y+1])

                if any(n != grid_np[x, y] for n in neighbors):
                    edges.append((x, y))

    return edges


def find_frontier(grid: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Find frontier cells (empty cells adjacent to filled cells).

    Args:
        grid: Grid tensor [H, W]

    Returns:
        List of (x, y) frontier coordinates
    """
    grid_np = grid.cpu().numpy()
    h, w = grid_np.shape
    frontier = []

    for x in range(h):
        for y in range(w):
            if grid_np[x, y] == 0:  # Background
                # Check if adjacent to non-background
                neighbors = []
                if x > 0: neighbors.append(grid_np[x-1, y])
                if x < h-1: neighbors.append(grid_np[x+1, y])
                if y > 0: neighbors.append(grid_np[x, y-1])
                if y < w-1: neighbors.append(grid_np[x, y+1])

                if any(n != 0 for n in neighbors):
                    frontier.append((x, y))

    return frontier


def find_symmetry_points(grid: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Find points along symmetry axes.

    Args:
        grid: Grid tensor [H, W]

    Returns:
        List of (x, y) coordinates on symmetry axes
    """
    h, w = grid.shape
    points = []

    # Horizontal center line
    mid_x = h // 2
    for y in range(0, w, 4):
        points.append((mid_x, y))

    # Vertical center line
    mid_y = w // 2
    for x in range(0, h, 4):
        points.append((x, mid_y))

    # Diagonal (if square)
    if h == w:
        for i in range(0, h, 4):
            points.append((i, i))
            points.append((i, h - i - 1))

    return points


def measure_progress(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> float:
    """
    Measure partial progress toward target (for shaped rewards).

    Args:
        pred_grid: Predicted grid [H, W]
        target_grid: Target grid [H, W]

    Returns:
        Progress score in [0.0, 1.0]
    """
    pred_np = pred_grid.cpu().numpy()
    target_np = target_grid.cpu().numpy()

    # 1. Correct color count (how many colors match?)
    pred_colors = set(pred_np.flatten())
    target_colors = set(target_np.flatten())
    color_match = len(pred_colors & target_colors) / max(len(target_colors), 1)

    # 2. Correct object count
    pred_objs = len(detect_objects(pred_grid))
    target_objs = len(detect_objects(target_grid))
    obj_match = 1.0 - abs(pred_objs - target_objs) / max(target_objs, 1)

    # 3. Spatial correlation (are things in roughly the right place?)
    correlation = np.corrcoef(pred_np.flatten(), target_np.flatten())[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    spatial_score = max(correlation, 0.0)

    return 0.4 * color_match + 0.3 * obj_match + 0.3 * spatial_score


def object_level_accuracy(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> float:
    """
    Compare grids at object level (did we get the right shapes?).

    Args:
        pred_grid: Predicted grid [H, W]
        target_grid: Target grid [H, W]

    Returns:
        Object-level accuracy in [0.0, 1.0]
    """
    pred_objs = detect_objects(pred_grid)
    target_objs = detect_objects(target_grid)

    if len(target_objs) == 0:
        return 1.0 if len(pred_objs) == 0 else 0.0

    # Count matching objects (IoU > 0.5)
    matches = 0
    for target_obj in target_objs:
        best_iou = 0.0
        for pred_obj in pred_objs:
            intersection = len(target_obj & pred_obj)
            union = len(target_obj | pred_obj)
            iou = intersection / union if union > 0 else 0.0
            best_iou = max(best_iou, iou)

        if best_iou > 0.5:
            matches += 1

    return matches / len(target_objs)


def structural_similarity(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> float:
    """
    SSIM-like metric for discrete grids.

    Args:
        pred_grid: Predicted grid [H, W]
        target_grid: Target grid [H, W]

    Returns:
        Structural similarity in [0.0, 1.0]
    """
    pred_np = pred_grid.cpu().numpy().astype(float)
    target_np = target_grid.cpu().numpy().astype(float)

    # Local statistics (3x3 windows)
    from scipy.ndimage import uniform_filter

    mu_pred = uniform_filter(pred_np, size=3)
    mu_target = uniform_filter(target_np, size=3)

    sigma_pred = uniform_filter(pred_np ** 2, size=3) - mu_pred ** 2
    sigma_target = uniform_filter(target_np ** 2, size=3) - mu_target ** 2
    sigma_pred_target = uniform_filter(pred_np * target_np, size=3) - mu_pred * mu_target

    c1 = 0.01
    c2 = 0.03

    ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred + sigma_target + c2))

    return float(np.mean(ssim))
