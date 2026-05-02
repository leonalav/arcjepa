import argparse
from pathlib import Path

from src.search.budget import SearchBudget
from src.search.miner import WinMiner


def load_game_ids(args):
    ids = list(args.game_id or [])
    if args.game_ids_file:
        ids.extend([line.strip() for line in Path(args.game_ids_file).read_text().splitlines() if line.strip() and not line.startswith("#")])
    if not ids:
        raise SystemExit("Provide --game_id or --game_ids_file")
    return ids


def main():
    parser = argparse.ArgumentParser(description="Mine verified ARC-AGI-3 winning recordings")
    parser.add_argument("--game_ids_file")
    parser.add_argument("--game_id", action="append")
    parser.add_argument("--out_dir", default="data/raw_episodes")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--algorithm", choices=["random_legal", "beam", "portfolio"], default="random_legal")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--episodes_per_game", type=int, default=1000)
    parser.add_argument("--max_nodes", type=int, default=10000)
    parser.add_argument("--max_env_steps", type=int, default=100000)
    parser.add_argument("--max_wallclock_sec", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_dir", default=None)
    args = parser.parse_args()

    budget = SearchBudget(
        max_steps=args.max_steps,
        max_nodes=args.max_nodes,
        max_env_steps=args.max_env_steps,
        max_wallclock_sec=args.max_wallclock_sec,
        episodes_per_game=args.episodes_per_game,
        seed=args.seed,
    )
    miner = WinMiner(
        game_ids=load_game_ids(args),
        out_dir=args.out_dir,
        algorithm=args.algorithm,
        workers=args.workers,
        budget=budget,
        seed=args.seed,
        env_dir=args.env_dir,
    )
    summary = miner.run()
    print(f"Episodes: {summary.episodes}")
    print(f"Wins: {summary.wins}")
    print(f"Valid action rate: {summary.valid_action_rate:.3f}")
    print(f"Manifest: {summary.manifest_path}")
    if summary.valid_action_rate < 0.99:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
