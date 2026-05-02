import random


def split_games(game_ids, val_fraction=0.2, test_fraction=0.0, seed=0):
    ids = sorted(set(game_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test = set(ids[:n_test])
    val = set(ids[n_test:n_test + n_val])
    train = set(ids[n_test + n_val:])
    return train, val, test


def episode_weight(row, mode="pretrain"):
    if mode == "expert_only":
        return 1.0 if row.get("success") else 0.0
    if mode == "mixed":
        return 2.0 if row.get("success") else 0.5
    return 1.0
