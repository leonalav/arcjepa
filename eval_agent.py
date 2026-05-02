import argparse
import json
from pathlib import Path

from src.evaluation.arc_eval import compute_rhae_like_score, rollout_metrics
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
    parser = argparse.ArgumentParser(description="Evaluate an ARC agent in the real environment")
    parser.add_argument("--game_ids_file")
    parser.add_argument("--game_id", action="append")
    parser.add_argument("--checkpoint")
    parser.add_argument("--agent", choices=["random", "heuristic", "model", "portfolio"], default="random")
    parser.add_argument("--search", choices=["random_legal", "beam", "portfolio"], default="random_legal")
    parser.add_argument("--episodes_per_game", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out_dir", default="data/eval_agent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_dir", default=None)
    args = parser.parse_args()
    budget = SearchBudget(max_steps=args.max_steps, episodes_per_game=args.episodes_per_game, seed=args.seed)
    miner = WinMiner(load_game_ids(args), args.out_dir, algorithm=args.search, workers=args.workers, budget=budget, seed=args.seed, env_dir=args.env_dir)
    summary = miner.run()
    records = []
    for result in summary.results:
        records.append({
            "game_family": result["game_id"].split("-", 1)[0],
            "success": result["success"],
            "valid_action_rate": result["valid_action_rate"],
            "steps": result["steps"],
            "rhae_like": compute_rhae_like_score(result["success"], result["steps"]),
        })
    metrics = rollout_metrics(records)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
