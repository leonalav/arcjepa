import concurrent.futures
import multiprocessing
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

import arc_agi
from torch.utils.data import Dataset
from arcengine import GameAction, GameState

from .arc_schema import (
    DEFAULT_NUM_ACTIONS,
    action_uses_coordinates,
    encode_available_actions,
    extract_first,
    make_coord_mask,
    parse_action_name,
    stable_game_family_id,
    stable_game_id,
)


def _extract_grid_static(obj):
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], list):
        return obj
    if isinstance(obj, dict):
        for k in ['grid', 'frame', 'board', 'state', 'observation', 'obs']:
            if k in obj:
                res = _extract_grid_static(obj[k])
                if res is not None:
                    return res
        for v in obj.values():
            res = _extract_grid_static(v)
            if res is not None:
                return res
    return None


def _extract_game_id(frame_data: Dict[str, Any], file_path: Optional[str] = None) -> Optional[str]:
    game_id = extract_first(frame_data, ['game_id', 'gameId', 'environment_id', 'env_id', 'task_id'])
    if game_id is not None:
        return str(game_id)
    meta = frame_data.get('metadata') or frame_data.get('meta') or {}
    game_id = extract_first(meta, ['game_id', 'gameId', 'environment_id', 'env_id', 'task_id'])
    if game_id is not None:
        return str(game_id)
    if file_path:
        stem = Path(file_path).stem
        parts = stem.split('_', 1)[0]
        if '-' in parts:
            return parts
    return None


def _extract_action_data(frame_data: Dict[str, Any]) -> Any:
    return extract_first(frame_data, ['action_input', 'action', 'action_data', 'input'], {})


def _extract_available_actions(frame_data: Dict[str, Any]) -> Any:
    raw = extract_first(frame_data, ['available_actions', 'availableActions', 'valid_actions', 'validActions'])
    if raw is not None:
        return raw
    obs = frame_data.get('observation') or frame_data.get('obs') or frame_data.get('state') or {}
    if isinstance(obs, dict):
        return extract_first(obs, ['available_actions', 'availableActions', 'valid_actions', 'validActions'])
    return None


def _terminal_success(frame_data: Dict[str, Any]) -> tuple[bool, bool]:
    state = extract_first(frame_data, ['terminal_state', 'state', 'status', 'game_state'])
    terminal = bool(extract_first(frame_data, ['terminal', 'done', 'is_terminal'], False))
    success = bool(extract_first(frame_data, ['success', 'win', 'won'], False))
    if hasattr(state, 'name'):
        state = state.name
    if state is not None:
        state_text = str(state).upper()
        terminal = terminal or state_text in {'WIN', 'GAME_OVER', 'LOSE', 'LOSS', 'DONE', 'TERMINAL'}
        success = success or state_text == 'WIN'
    return terminal, success


def _score_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _preprocess_frame_static(
    frame_data: Dict[str, Any],
    max_grid_size: int,
    file_path: Optional[str] = None,
    num_actions: int = DEFAULT_NUM_ACTIONS,
    max_games: int = 4096,
    max_game_families: int = 512,
) -> Dict[str, Any]:
    grid_data = _extract_grid_static(frame_data) or []
    grid = np.array(grid_data, dtype=np.int64)

    while grid.ndim > 2:
        grid = grid[0]
    if grid.ndim < 2 or grid.size == 0:
        grid = np.zeros((1, 1), dtype=np.int64)

    if grid.min(initial=0) < 0 or grid.max(initial=0) > 15:
        raise ValueError(f"ARC grid colors must be in [0, 15], got [{grid.min()}, {grid.max()}]")

    h, w = grid.shape
    safe_h, safe_w = min(h, max_grid_size), min(w, max_grid_size)
    padded_grid = np.zeros((max_grid_size, max_grid_size), dtype=np.int64)
    padded_grid[:safe_h, :safe_w] = grid[:safe_h, :safe_w]

    action_data = _extract_action_data(frame_data)
    if isinstance(action_data, str):
        action_type = action_data
        ax, ay = 0, 0
    elif isinstance(action_data, dict):
        action_type = action_data.get('action', action_data.get('type', 'NONE'))
        ax = max(0, min(int(action_data.get('x', 0)), max_grid_size - 1))
        ay = max(0, min(int(action_data.get('y', 0)), max_grid_size - 1))
    else:
        action_type = action_data
        ax, ay = 0, 0

    action_idx = parse_action_name(action_type, num_actions=num_actions, strict=False)
    available_mask = encode_available_actions(_extract_available_actions(frame_data), num_actions=num_actions)
    if action_idx > 0 and not available_mask[action_idx]:
        available_mask[action_idx] = True

    game_id = _extract_game_id(frame_data, file_path)
    terminal, success = _terminal_success(frame_data)
    score = _score_value(extract_first(frame_data, ['score', 'reward', 'rhae', 'RHAE'], 0.0))
    step_index = int(extract_first(frame_data, ['step', 'step_index', 'timestep', 'frame_index'], 0) or 0)
    episode_success = bool(extract_first(frame_data, ['episode_success'], success))
    return_to_go = _score_value(extract_first(frame_data, ['return_to_go'], 1.0 if episode_success else 0.0))
    distance_to_win = _score_value(extract_first(frame_data, ['distance_to_win'], 0.0 if episode_success else 1.0))
    steps_to_win = int(extract_first(frame_data, ['steps_to_win'], max(1, step_index + 1)) or 1)
    efficiency_target = _score_value(extract_first(frame_data, ['efficiency_target'], (1.0 / max(1, steps_to_win)) if episode_success else 0.0))
    mcts_visit_policy = extract_first(frame_data, ['mcts_visit_policy', 'visit_policy', 'visit_counts'], None)

    return {
        'grid': torch.from_numpy(padded_grid),
        'action_type': action_idx,
        'x': ax,
        'y': ay,
        'available_actions_mask': available_mask,
        'coord_mask': action_uses_coordinates(action_idx),
        'game_id': stable_game_id(game_id, max_games),
        'game_family': stable_game_family_id(game_id, max_game_families),
        'terminal': terminal,
        'success': success,
        'score': score,
        'step_index': step_index,
        'episode_success': episode_success,
        'return_to_go': return_to_go,
        'distance_to_win': distance_to_win,
        'steps_to_win': steps_to_win,
        'efficiency_target': efficiency_target,
        'mcts_visit_policy': mcts_visit_policy,
    }


def _process_single_file(args):
    file_path, window_size, stride, max_grid_size, filter_noops, num_actions, max_games, max_game_families = args
    rows = {k: [] for k in [
        'grids', 'actions', 'xs', 'ys', 'available_masks', 'coord_masks', 'game_ids',
        'game_families', 'terminals', 'successes', 'scores', 'step_indices',
        'episode_successes', 'returns_to_go', 'distances_to_win', 'steps_to_win',
        'efficiency_targets', 'mcts_visit_policies'
    ]}

    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            frame_data = json.loads(line)
            processed = _preprocess_frame_static(
                frame_data,
                max_grid_size,
                file_path=file_path,
                num_actions=num_actions,
                max_games=max_games,
                max_game_families=max_game_families,
            )
            rows['grids'].append(processed['grid'].to(torch.uint8))
            rows['actions'].append(processed['action_type'])
            rows['xs'].append(processed['x'])
            rows['ys'].append(processed['y'])
            rows['available_masks'].append(processed['available_actions_mask'])
            rows['coord_masks'].append(processed['coord_mask'])
            rows['game_ids'].append(processed['game_id'])
            rows['game_families'].append(processed['game_family'])
            rows['terminals'].append(processed['terminal'])
            rows['successes'].append(processed['success'])
            rows['scores'].append(processed['score'])
            rows['step_indices'].append(processed['step_index'])
            rows['episode_successes'].append(processed['episode_success'])
            rows['returns_to_go'].append(processed['return_to_go'])
            rows['distances_to_win'].append(processed['distance_to_win'])
            rows['steps_to_win'].append(processed['steps_to_win'])
            rows['efficiency_targets'].append(processed['efficiency_target'])
            rows['mcts_visit_policies'].append(processed['mcts_visit_policy'])

    traj_len = len(rows['grids'])
    if traj_len < 2:
        return None

    grids_tensor = torch.stack(rows['grids'])
    actions_tensor = torch.tensor(rows['actions'], dtype=torch.long)
    xs_tensor = torch.tensor(rows['xs'], dtype=torch.long)
    ys_tensor = torch.tensor(rows['ys'], dtype=torch.long)
    available_tensor = torch.stack(rows['available_masks']).bool()
    coord_tensor = torch.tensor(rows['coord_masks'], dtype=torch.bool)
    game_ids_tensor = torch.tensor(rows['game_ids'], dtype=torch.long)
    game_families_tensor = torch.tensor(rows['game_families'], dtype=torch.long)
    terminals_tensor = torch.tensor(rows['terminals'], dtype=torch.bool)
    successes_tensor = torch.tensor(rows['successes'], dtype=torch.bool)
    scores_tensor = torch.tensor(rows['scores'], dtype=torch.float32)
    step_indices_tensor = torch.tensor(rows['step_indices'], dtype=torch.long)
    episode_successes_tensor = torch.tensor(rows['episode_successes'], dtype=torch.bool)
    returns_to_go_tensor = torch.tensor(rows['returns_to_go'], dtype=torch.float32)
    distances_to_win_tensor = torch.tensor(rows['distances_to_win'], dtype=torch.float32)
    steps_to_win_tensor = torch.tensor(rows['steps_to_win'], dtype=torch.long)
    efficiency_targets_tensor = torch.tensor(rows['efficiency_targets'], dtype=torch.float32)
    if any(v is not None for v in rows['mcts_visit_policies']):
        policy_rows = [v if v is not None else [0.0] * num_actions for v in rows['mcts_visit_policies']]
        mcts_visit_policies_tensor = torch.tensor(policy_rows, dtype=torch.float32)
    else:
        mcts_visit_policies_tensor = torch.zeros(traj_len, num_actions, dtype=torch.float32)

    valid_starts = []
    for i in range(0, traj_len - window_size, stride):
        if filter_noops:
            chunk_grids = grids_tensor[i : i + window_size + 1]
            if torch.all(chunk_grids == chunk_grids[0]):
                continue
        valid_starts.append(i)

    return (
        grids_tensor, actions_tensor, xs_tensor, ys_tensor, available_tensor, coord_tensor,
        game_ids_tensor, game_families_tensor, terminals_tensor, successes_tensor,
        scores_tensor, step_indices_tensor, episode_successes_tensor, returns_to_go_tensor,
        distances_to_win_tensor, steps_to_win_tensor, efficiency_targets_tensor,
        mcts_visit_policies_tensor, valid_starts
    )


class ARCTrajectoryDataset(Dataset):
    def __init__(
        self,
        recording_files: List[str],
        window_size: int = 24,
        stride: int = 4,
        max_grid_size: int = 64,
        multistep_k: int = 1,
        compute_temporal_masks: bool = False,
        filter_noops: bool = False,
        num_actions: int = DEFAULT_NUM_ACTIONS,
        max_games: int = 4096,
        max_game_families: int = 512,
    ):
        self.window_size = window_size
        self.stride = stride
        self.max_grid_size = max_grid_size
        self.multistep_k = multistep_k
        self.compute_temporal_masks = compute_temporal_masks
        self.filter_noops = filter_noops
        self.num_actions = num_actions
        self.max_games = max_games
        self.max_game_families = max_game_families
        self.chunks = []
        self.trajectories = []
        self._load_recordings(recording_files)

    def _load_recordings(self, recording_files: List[str]):
        max_workers = min(12, multiprocessing.cpu_count())
        args_list = [
            (f, self.window_size, self.stride, self.max_grid_size, self.filter_noops,
             self.num_actions, self.max_games, self.max_game_families)
            for f in recording_files
        ]
        ctx = multiprocessing.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            results = list(executor.map(_process_single_file, args_list))

        for res in results:
            if res is None:
                continue
            valid_starts = res[-1]
            traj_idx = len(self.trajectories)
            for start_idx in valid_starts:
                self.chunks.append((traj_idx, start_idx))
            self.trajectories.append(res[:-1])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        traj_idx, start_idx = self.chunks[idx]
        end_idx = start_idx + self.window_size + 1
        (
            grids, actions, xs, ys, available, coord_masks, game_ids, game_families,
            terminals, successes, scores, step_indices, episode_successes, returns_to_go,
            distances_to_win, steps_to_win, efficiency_targets, mcts_visit_policies
        ) = [t[start_idx:end_idx] for t in self.trajectories[traj_idx]]

        state_changed_mask = self._compute_temporal_mask(grids[:-1].long(), grids[1:].long())
        seq_mask = torch.ones(self.window_size, dtype=torch.float32)
        efficiency = efficiency_targets[:-1].float()

        result = {
            'states': grids[:-1].long(),
            'actions': actions[:-1].long(),
            'coords_x': xs[:-1].long(),
            'coords_y': ys[:-1].long(),
            'target_states': grids[1:].long(),
            'final_state': grids[-1].long(),
            'available_actions_mask': available[:-1].bool(),
            'coord_mask': coord_masks[:-1].bool(),
            'game_id': game_ids[:-1].long(),
            'game_family': game_families[:-1].long(),
            'terminal': terminals[:-1].float(),
            'success': successes[:-1].float(),
            'score': scores[:-1].float(),
            'step_index': step_indices[:-1].long(),
            'efficiency_target': efficiency.float(),
            'episode_success': episode_successes[:-1].float(),
            'return_to_go': returns_to_go[:-1].float(),
            'distance_to_win': distances_to_win[:-1].float(),
            'steps_to_win': steps_to_win[:-1].long(),
            'mcts_visit_policy': mcts_visit_policies[:-1].float(),
            'seq_mask': seq_mask,
            'state_changed_mask': state_changed_mask.float(),
        }
        if self.compute_temporal_masks:
            result['temporal_mask'] = state_changed_mask.float()
        return result

    def _efficiency_targets(self, successes: torch.Tensor, step_indices: torch.Tensor) -> torch.Tensor:
        steps = step_indices.float().clamp_min(0.0)
        return successes.float() / (steps + 1.0)

    def _compute_temporal_mask(self, states: torch.Tensor, target_states: torch.Tensor) -> torch.Tensor:
        return (states != target_states).float()

    def validate_arc_compliance(self):
        print(f"Validating {len(self.chunks)} chunks for ARC-AGI-3 compliance...")
        for chunk_idx, (traj_idx, start_idx) in enumerate(self.chunks):
            end_idx = start_idx + self.window_size + 1
            grids, actions, xs, ys, available, *_ = [t[start_idx:end_idx] for t in self.trajectories[traj_idx]]
            for step_idx, grid in enumerate(grids):
                assert grid.shape == (64, 64), f"Chunk {chunk_idx}, Step {step_idx}: Invalid grid shape {grid.shape}"
                assert grid.min() >= 0 and grid.max() <= 15, f"Chunk {chunk_idx}, Step {step_idx}: Invalid color range [{grid.min()}, {grid.max()}]"
                if step_idx < len(actions) - 1:
                    action = int(actions[step_idx])
                    assert 0 <= action < self.num_actions, f"Chunk {chunk_idx}, Step {step_idx}: Invalid action type {action}"
                    assert 0 <= xs[step_idx] < 64 and 0 <= ys[step_idx] < 64, f"Chunk {chunk_idx}, Step {step_idx}: Invalid coords ({xs[step_idx]}, {ys[step_idx]})"
                    assert available[step_idx].shape[0] == self.num_actions, f"Chunk {chunk_idx}, Step {step_idx}: Invalid action mask shape"
                    assert action == 0 or bool(available[step_idx, action]), f"Chunk {chunk_idx}, Step {step_idx}: Target action {action} unavailable"
        print(f"Dataset validation passed: {len(self.chunks)} chunks comply with ARC-AGI-3 specs")


class FastHFARCDataset(Dataset):
    def __init__(
        self,
        hf_repo_id: str,
        split: str = "train",
        compute_temporal_masks: bool = False,
        max_seq_len: int = 64,
        num_actions: int = DEFAULT_NUM_ACTIONS,
        max_games: int = 4096,
        max_game_families: int = 512,
    ):
        super().__init__()
        self.hf_ds = load_dataset(hf_repo_id, split=split)
        self.compute_temporal_masks = compute_temporal_masks
        self.max_seq_len = max_seq_len
        self.num_actions = num_actions
        self.max_games = max_games
        self.max_game_families = max_game_families

    def __len__(self):
        return len(self.hf_ds)

    def _pad_or_window(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        current_len = tensor.shape[0]
        if current_len == target_len:
            return tensor
        if current_len > target_len:
            start = random.randint(0, current_len - target_len)
            return tensor[start:start + target_len]
        pad_shape = list(tensor.shape)
        pad_shape[0] = target_len - current_len
        padding = tensor[-1:].expand(pad_shape)
        return torch.cat([tensor, padding], dim=0)

    def _item_tensor(self, item: Dict[str, Any], key: str, default: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if key in item and item[key] is not None:
            return torch.tensor(item[key], dtype=dtype)
        return default.to(dtype=dtype)

    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        T = self.max_seq_len
        states_raw = torch.tensor(item['states'], dtype=torch.long)
        actions_raw = torch.tensor(item['actions'], dtype=torch.long)
        coords_x_raw = torch.tensor(item['coords_x'], dtype=torch.long)
        coords_y_raw = torch.tensor(item['coords_y'], dtype=torch.long)
        target_states_raw = torch.tensor(item['target_states'], dtype=torch.long)

        raw_len = states_raw.shape[0]
        states = self._pad_or_window(states_raw, T)
        actions = self._pad_or_window(actions_raw, T)
        coords_x = self._pad_or_window(coords_x_raw, T)
        coords_y = self._pad_or_window(coords_y_raw, T)
        target_states = self._pad_or_window(target_states_raw, T)
        valid_len = min(raw_len, T)
        seq_mask = torch.zeros(T, dtype=torch.float32)
        seq_mask[:valid_len] = 1.0

        available_default = torch.ones(raw_len, self.num_actions, dtype=torch.bool)
        available_default[:, 0] = False
        available_raw = self._item_tensor(item, 'available_actions_mask', available_default, torch.bool)
        available = self._pad_or_window(available_raw, T).bool()
        action_positions = actions.clamp(0, self.num_actions - 1).unsqueeze(-1)
        available.scatter_(1, action_positions, True)

        coord_default = make_coord_mask(actions_raw)
        coord_mask = self._pad_or_window(self._item_tensor(item, 'coord_mask', coord_default, torch.bool), T).bool()
        game_id = self._pad_or_window(self._item_tensor(item, 'game_id', torch.zeros(raw_len, dtype=torch.long), torch.long), T).long()
        game_family = self._pad_or_window(self._item_tensor(item, 'game_family', torch.zeros(raw_len, dtype=torch.long), torch.long), T).long()
        terminal = self._pad_or_window(self._item_tensor(item, 'terminal', torch.zeros(raw_len), torch.float32), T).float()
        success = self._pad_or_window(self._item_tensor(item, 'success', torch.zeros(raw_len), torch.float32), T).float()
        score = self._pad_or_window(self._item_tensor(item, 'score', torch.zeros(raw_len), torch.float32), T).float()
        step_index = self._pad_or_window(self._item_tensor(item, 'step_index', torch.arange(raw_len), torch.long), T).long()
        episode_success = self._pad_or_window(self._item_tensor(item, 'episode_success', success, torch.float32), T).float()
        return_to_go = self._pad_or_window(self._item_tensor(item, 'return_to_go', episode_success, torch.float32), T).float()
        distance_to_win = self._pad_or_window(self._item_tensor(item, 'distance_to_win', 1.0 - episode_success, torch.float32), T).float()
        steps_to_win = self._pad_or_window(self._item_tensor(item, 'steps_to_win', torch.full((raw_len,), T + 1), torch.long), T).long()
        efficiency = self._pad_or_window(self._item_tensor(item, 'efficiency_target', episode_success / steps_to_win.float().clamp_min(1.0), torch.float32), T).float()
        visit_default = torch.zeros(raw_len, self.num_actions, dtype=torch.float32)
        mcts_visit_policy = self._pad_or_window(self._item_tensor(item, 'mcts_visit_policy', visit_default, torch.float32), T).float()
        state_changed_mask = (states != target_states).float()

        result = {
            'states': states,
            'actions': actions,
            'coords_x': coords_x,
            'coords_y': coords_y,
            'target_states': target_states,
            'final_state': target_states[-1],
            'available_actions_mask': available,
            'coord_mask': coord_mask,
            'game_id': game_id,
            'game_family': game_family,
            'terminal': terminal,
            'success': success,
            'score': score,
            'step_index': step_index,
            'efficiency_target': efficiency,
            'episode_success': episode_success,
            'return_to_go': return_to_go,
            'distance_to_win': distance_to_win,
            'steps_to_win': steps_to_win,
            'mcts_visit_policy': mcts_visit_policy,
            'seq_mask': seq_mask,
            'state_changed_mask': state_changed_mask,
        }
        if self.compute_temporal_masks:
            result['temporal_mask'] = state_changed_mask
        return result


def create_mock_trajectory(
    output_dir: str,
    num_trajectories: int = 1000,
    min_state_changes: int = 5,
    min_fg_ratio: float = 0.1,
    max_attempts: int = 5000,
    split_seed: int = 0,
):
    from .heuristic_policy import ARCHeuristicPolicy
    import shutil

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    env_dir = PROJECT_ROOT / "data" / "arc_environments"
    env_dir.mkdir(parents=True, exist_ok=True)
    arc = arc_agi.Arcade(environments_dir=str(env_dir))

    game_ids = [
        "vc33-5430563c", "sp80-589a99af", "re86-8af5384d", "m0r0-492f87ba",
        "wa30-ee6fef47", "r11l-495a7899", "tu93-0768757b", "ka59-38d34dbb",
        "sc25-635fd71a", "ft09-0d8bbf25", "ls20-9607627b", "ar25-0c556536",
        "lf52-271a04aa", "g50t-5849a774", "su15-1944f8ab", "tn36-ef4dde99",
        "sb26-7fbdac44", "cn04-2fe56bfb", "lp85-305b61c3", "s5i5-18d95033",
        "sk48-d8078629", "cd82-fb555c5d", "tr87-cd924810", "bp35-0a0ad940",
        "dc22-fdcac232"
    ]
    rng = random.Random(split_seed)
    rng.shuffle(game_ids)
    val_split_size = min(5, len(game_ids) // 5)
    val_games = set(game_ids[:val_split_size])

    policy = ARCHeuristicPolicy(exploration_rate=0.2)
    successful = 0
    attempts = 0
    valid_action_count = 0
    total_action_count = 0
    terminal_count = 0
    win_count = 0

    def get_grid(obs):
        grid_data = None
        if hasattr(obs, 'grid'):
            grid_data = obs.grid
        elif hasattr(obs, 'frame'):
            grid_data = obs.frame
        elif hasattr(obs, 'board'):
            grid_data = obs.board
        if grid_data is not None:
            arr = np.array(grid_data, dtype=np.int64)
            while arr.ndim > 2:
                arr = arr[0]
            if arr.ndim < 2:
                arr = np.zeros((2, 2), dtype=np.int64)
            return arr
        return None

    def get_available(obs):
        for name in ('available_actions', 'availableActions', 'valid_actions', 'validActions'):
            if hasattr(obs, name):
                return getattr(obs, name)
        return None

    while successful < num_trajectories and attempts < max_attempts:
        game_id = game_ids[attempts % len(game_ids)]
        try:
            env = arc.make(game_id, save_recording=True)
            obs = env.reset()
            current_grid = get_grid(obs)
            if obs is None or current_grid is None:
                attempts += 1
                continue
            state_changes = 0
            prev_grid = current_grid.copy()
            max_fg_ratio = 0.0
            step = 0
            max_steps = 100
            while step < max_steps:
                if obs is None or current_grid is None:
                    break
                available = get_available(obs)
                action_enum, x, y = policy.select_action(current_grid, step, available_actions=available, game_id=game_id)
                total_action_count += 1
                if available is None or parse_action_name(action_enum) in set(encode_available_actions(available).nonzero(as_tuple=False).flatten().tolist()):
                    valid_action_count += 1
                if action_enum == GameAction.ACTION6:
                    obs = env.step(action_enum, data={"x": x, "y": y})
                else:
                    obs = env.step(action_enum)
                current_grid = get_grid(obs)
                if obs and current_grid is not None:
                    if not np.array_equal(current_grid, prev_grid):
                        state_changes += 1
                    max_fg_ratio = max(max_fg_ratio, np.mean(current_grid != 0))
                    prev_grid = current_grid.copy()
                if obs and obs.state in [GameState.WIN, GameState.GAME_OVER]:
                    terminal_count += 1
                    if obs.state == GameState.WIN:
                        win_count += 1
                    break
                step += 1
            if state_changes >= min_state_changes and max_fg_ratio >= min_fg_ratio:
                successful += 1
                if successful % 100 == 0:
                    print(f"Progress: {successful}/{num_trajectories} valid trajectories")
                    print(f"  Last trajectory: {state_changes} changes, {max_fg_ratio:.3f} fg_ratio")
        except Exception as e:
            print(f"Warning: Failed to generate trajectory {attempts}: {e}")
        attempts += 1

    local_recordings = Path("recordings")
    if local_recordings.exists():
        train_dir = Path(output_dir) / "train"
        val_dir = Path(output_dir) / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        moved_train = 0
        moved_val = 0
        for jsonl_file in local_recordings.glob("**/*.jsonl"):
            file_game_id = next((gid for gid in game_ids if jsonl_file.name.startswith(gid)), None)
            target_dir = val_dir if file_game_id in val_games else train_dir
            new_name = f"{jsonl_file.stem}_{moved_train + moved_val}.jsonl"
            shutil.copy(jsonl_file, target_dir / new_name)
            if file_game_id in val_games:
                moved_val += 1
            else:
                moved_train += 1
        print(f"\nSuccessfully split dataset: {moved_train} train, {moved_val} val")

    valid_rate = valid_action_count / max(1, total_action_count)
    print(f"\nGeneration complete:")
    print(f"  Valid trajectories: {successful}/{num_trajectories}")
    print(f"  Total attempts: {attempts}")
    print(f"  Valid action rate: {valid_rate:.3f}")
    print(f"  Terminal rate: {terminal_count / max(1, attempts):.3f}")
    print(f"  Win rate: {win_count / max(1, attempts):.3f}")
    return successful
