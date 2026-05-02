import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="World-model pretraining wrapper")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="checkpoints_world")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    cmd = [
        sys.executable, "train.py",
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--policy_weight", "0.05",
        "--coord_weight", "0.05",
        "--value_weight", "0.05",
        "--efficiency_weight", "0.05",
    ] + args.extra
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
