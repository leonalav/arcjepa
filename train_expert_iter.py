import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print(" ".join(cmd))
    code = subprocess.call(cmd)
    if code != 0:
        raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser(description="ARC expert-iteration orchestrator")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--game_ids_file", required=True)
    parser.add_argument("--raw_dir", default="data/raw_episodes")
    parser.add_argument("--index_path", default="data/replay_index/episodes.sqlite")
    parser.add_argument("--shard_root", default="data/train_shards")
    parser.add_argument("--checkpoint_root", default="checkpoints_expert_iter")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--episodes_per_game", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()
    checkpoint = None
    for iteration in range(args.iterations):
        raw_iter = str(Path(args.raw_dir) / f"iter_{iteration:04d}")
        shard_iter = str(Path(args.shard_root) / f"mixed_v{iteration:04d}")
        ckpt_iter = str(Path(args.checkpoint_root) / f"iter_{iteration:04d}")
        run([sys.executable, "mine_wins.py", "--game_ids_file", args.game_ids_file, "--out_dir", raw_iter, "--workers", str(args.workers), "--algorithm", "portfolio", "--episodes_per_game", str(args.episodes_per_game), "--max_steps", str(args.max_steps)])
        run([sys.executable, "build_dataset.py", "--raw_dir", raw_iter, "--index_path", args.index_path, "--out_dir", shard_iter, "--mode", "mixed", "--overwrite"])
        train_cmd = [sys.executable, "train_policy.py", "--data_dir", shard_iter, "--output_dir", ckpt_iter]
        if checkpoint:
            train_cmd += ["--resume_checkpoint", checkpoint]
        run(train_cmd)
        checkpoint = str(Path(ckpt_iter) / "world_model_epoch_1.pt")


if __name__ == "__main__":
    main()
