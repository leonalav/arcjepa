import argparse

from src.data.dataset_builder import build_dataset_shards


def main():
    parser = argparse.ArgumentParser(description="Build ARC expert-iteration dataset shards")
    parser.add_argument("--raw_dir", default="data/raw_episodes")
    parser.add_argument("--index_path", default="data/replay_index/episodes.sqlite")
    parser.add_argument("--out_dir", default="data/train_shards/pretrain_v0001")
    parser.add_argument("--mode", choices=["pretrain", "expert_only", "mixed"], default="pretrain")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--test_fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build_dataset_shards(
        raw_dir=args.raw_dir,
        index_path=args.index_path,
        out_dir=args.out_dir,
        mode=args.mode,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        overwrite_index=args.overwrite,
    )
    print(manifest)


if __name__ == "__main__":
    main()
