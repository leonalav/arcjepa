import json
from pathlib import Path

from src.data.episode_reader import read_episode
from src.data.replay_index import build_index, query_episodes
from src.data.sampling import split_games


def enrich_rows(rows: list[dict], max_steps: int = 100, gamma: float | None = None) -> list[dict]:
    n = len(rows)
    success = bool(rows and rows[-1].get("success"))
    enriched = []
    for i, row in enumerate(rows):
        remaining = max(1, n - i)
        out = dict(row)
        out["episode_success"] = success
        out["steps_to_win"] = remaining if success else max_steps + 1
        out["distance_to_win"] = remaining / max(1, n) if success else 1.0
        out["return_to_go"] = (gamma ** (remaining - 1) if gamma is not None else 1.0) if success else 0.0
        out["efficiency_target"] = 1.0 / remaining if success else 0.0
        enriched.append(out)
    return enriched


def build_dataset_shards(
    raw_dir: str | Path,
    index_path: str | Path,
    out_dir: str | Path,
    mode: str = "pretrain",
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 0,
    overwrite_index: bool = False,
):
    build_index(raw_dir, index_path, overwrite=overwrite_index)
    episodes = query_episodes(index_path, mode=mode)
    train_games, val_games, test_games = split_games([e["game_id"] for e in episodes], val_fraction, test_fraction, seed)
    out_dir = Path(out_dir)
    counts = {"train": 0, "val": 0, "test": 0}
    files = {
        "train": out_dir / "train" / "episodes.jsonl",
        "val": out_dir / "val" / "episodes.jsonl",
        "test": out_dir / "test" / "episodes.jsonl",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    handles = {split: path.open("a", encoding="utf-8") for split, path in files.items()}
    try:
        for episode in episodes:
            game_id = episode["game_id"]
            split = "test" if game_id in test_games else "val" if game_id in val_games else "train"
            for row in enrich_rows(read_episode(episode["path"])):
                handles[split].write(json.dumps(row, separators=(",", ":")) + "\n")
                counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {
        "mode": mode,
        "counts": counts,
        "train_games": sorted(train_games),
        "val_games": sorted(val_games),
        "test_games": sorted(test_games),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
