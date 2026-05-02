from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


def compute_rhae_like_score(
    success: bool,
    steps: int,
    optimal_or_baseline_steps: Optional[float] = None,
    invalid_actions: int = 0,
) -> float:
    if not success:
        return 0.0
    steps = max(1, int(steps))
    baseline = max(1.0, float(optimal_or_baseline_steps or steps))
    efficiency = min(1.0, baseline / steps)
    invalid_penalty = 1.0 / (1.0 + max(0, invalid_actions))
    return efficiency * invalid_penalty


def summarize_by_game_family(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    groups = defaultdict(list)
    for record in records:
        groups[str(record.get('game_family', 'unknown'))].append(record)

    summary = {}
    for family, rows in groups.items():
        n = max(1, len(rows))
        summary[family] = {
            'episodes': float(n),
            'success_rate': sum(float(r.get('success', False)) for r in rows) / n,
            'valid_action_rate': sum(float(r.get('valid_action_rate', 0.0)) for r in rows) / n,
            'avg_steps': sum(float(r.get('steps', 0.0)) for r in rows) / n,
            'rhae_like': sum(float(r.get('rhae_like', 0.0)) for r in rows) / n,
        }
    return summary


def rollout_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = max(1, len(records))
    aggregate = {
        'episodes': len(records),
        'success_rate': sum(float(r.get('success', False)) for r in records) / n,
        'valid_action_rate': sum(float(r.get('valid_action_rate', 0.0)) for r in records) / n,
        'avg_steps': sum(float(r.get('steps', 0.0)) for r in records) / n,
        'rhae_like': sum(float(r.get('rhae_like', 0.0)) for r in records) / n,
    }
    aggregate['by_game_family'] = summarize_by_game_family(records)
    return aggregate
