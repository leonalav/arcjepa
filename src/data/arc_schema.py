import hashlib
import re
from typing import Any, Iterable, Optional, Sequence

import torch


NONE_ACTION = "NONE"
SUBMIT_ACTION = "SUBMIT"
DEFAULT_ACTION_NAMES = [NONE_ACTION] + [f"ACTION{i}" for i in range(1, 9)] + [SUBMIT_ACTION]
DEFAULT_NUM_ACTIONS = len(DEFAULT_ACTION_NAMES)
ACTION_TO_IDX = {name: idx for idx, name in enumerate(DEFAULT_ACTION_NAMES)}
IDX_TO_ACTION = {idx: name for name, idx in ACTION_TO_IDX.items()}
COORDINATE_ACTION_NAMES = {f"ACTION{i}" for i in range(1, 9)}


def stable_hash_id(value: Any, modulo: int) -> int:
    if value is None or modulo <= 1:
        return 0
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def game_family(game_id: Any) -> str:
    if game_id is None:
        return "unknown"
    text = str(game_id)
    return text.split("-", 1)[0] if "-" in text else text


def stable_game_id(game_id: Any, max_games: int = 4096) -> int:
    return stable_hash_id(game_id, max_games)


def stable_game_family_id(game_id: Any, max_families: int = 512) -> int:
    return stable_hash_id(game_family(game_id), max_families)


def _action_name_from_any(action: Any) -> str:
    if action is None:
        return NONE_ACTION
    if hasattr(action, "name"):
        return str(action.name).upper()
    if isinstance(action, dict):
        for key in ("action", "type", "name", "action_type"):
            if key in action:
                return _action_name_from_any(action[key])
        return NONE_ACTION
    text = str(action).strip().upper()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def parse_action_name(action: Any, *, num_actions: int = DEFAULT_NUM_ACTIONS, strict: bool = False) -> int:
    name = _action_name_from_any(action)
    if name in ACTION_TO_IDX:
        idx = ACTION_TO_IDX[name]
    else:
        match = re.fullmatch(r"ACTION(\d+)", name)
        if not match:
            if strict:
                raise ValueError(f"Unknown ARC action: {action!r}")
            return ACTION_TO_IDX[NONE_ACTION]
        idx = int(match.group(1))
    if idx >= num_actions:
        if strict:
            raise ValueError(f"ARC action {name} maps to {idx}, outside num_actions={num_actions}")
        return ACTION_TO_IDX[NONE_ACTION]
    return idx


def action_name(action_idx: int) -> str:
    return IDX_TO_ACTION.get(int(action_idx), f"ACTION{int(action_idx)}")


def action_uses_coordinates(action: Any) -> bool:
    name = action_name(action) if isinstance(action, int) else _action_name_from_any(action)
    return name in COORDINATE_ACTION_NAMES


def default_available_actions(num_actions: int = DEFAULT_NUM_ACTIONS, include_none: bool = False) -> torch.Tensor:
    mask = torch.ones(num_actions, dtype=torch.bool)
    if num_actions > 0 and not include_none:
        mask[0] = False
    return mask


def encode_available_actions(
    raw_available: Optional[Iterable[Any]],
    *,
    num_actions: int = DEFAULT_NUM_ACTIONS,
    fallback_all: bool = True,
) -> torch.Tensor:
    if raw_available is None:
        return default_available_actions(num_actions) if fallback_all else torch.zeros(num_actions, dtype=torch.bool)
    mask = torch.zeros(num_actions, dtype=torch.bool)
    for action in raw_available:
        idx = parse_action_name(action, num_actions=num_actions, strict=False)
        if 0 <= idx < num_actions:
            mask[idx] = True
    if not mask.any() and fallback_all:
        return default_available_actions(num_actions)
    return mask


def available_indices(mask_or_actions: Optional[Any], *, num_actions: int = DEFAULT_NUM_ACTIONS) -> list[int]:
    if mask_or_actions is None:
        return default_available_actions(num_actions).nonzero(as_tuple=False).flatten().tolist()
    if isinstance(mask_or_actions, torch.Tensor):
        return mask_or_actions.bool().nonzero(as_tuple=False).flatten().tolist()
    return encode_available_actions(mask_or_actions, num_actions=num_actions).nonzero(as_tuple=False).flatten().tolist()


def make_coord_mask(actions: torch.Tensor) -> torch.Tensor:
    flat = actions.reshape(-1).detach().cpu().tolist()
    values = [action_uses_coordinates(int(a)) for a in flat]
    return torch.tensor(values, dtype=torch.bool, device=actions.device).reshape(actions.shape)


def masked_action_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return logits
    mask = mask.to(device=logits.device, dtype=torch.bool)
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def extract_first(obj: Any, keys: Sequence[str], default: Any = None) -> Any:
    if not isinstance(obj, dict):
        return default
    for key in keys:
        if key in obj:
            return obj[key]
    return default
