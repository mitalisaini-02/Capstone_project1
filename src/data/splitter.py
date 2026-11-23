"""
Dataset splitting module ensuring strict leakage prevention.
Supports Vehicle-Disjoint (by Consumer ID) and Pair-Disjoint splitting.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
import random


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    metadata: List[Dict[str, Any]],
    strategy: str = "vehicle_disjoint_consumer",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]],
    Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]],
    Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]],
    Dict[str, Any]
]:
    """
    Splits dataset into Train, Validation, and Test sets without temporal or entity leakage.

    Strategies:
      - 'vehicle_disjoint_consumer': Partitions unique Consumer IDs into disjoint sets.
        Guarantees that no consumer appearing in test/val is ever seen during training.
      - 'disjoint_pairs': Partitions unique (provider, consumer) pair IDs into disjoint sets.

    Returns:
      (X_train, y_train, meta_train),
      (X_val, y_val, meta_val),
      (X_test, y_test, meta_test),
      split_info_dict
    """
    if len(metadata) == 0:
        raise ValueError("Cannot split an empty dataset.")

    rng = random.Random(seed)

    if strategy == "vehicle_disjoint_consumer":
        unique_consumers = sorted(list(set(m["consumer_id"] for m in metadata)))
        rng.shuffle(unique_consumers)

        n_c = len(unique_consumers)
        n_train = max(1, int(round(n_c * train_ratio)))
        n_val = max(1, int(round(n_c * val_ratio)))
        # Remainder goes to test
        train_consumers = set(unique_consumers[:n_train])
        val_consumers = set(unique_consumers[n_train:n_train + n_val])
        test_consumers = set(unique_consumers[n_train + n_val:])

        # If test set is empty due to small consumer count, rebalance
        if not test_consumers and len(val_consumers) > 1:
            test_consumers = {val_consumers.pop()}

        train_indices = [i for i, m in enumerate(metadata) if m["consumer_id"] in train_consumers]
        val_indices = [i for i, m in enumerate(metadata) if m["consumer_id"] in val_consumers]
        test_indices = [i for i, m in enumerate(metadata) if m["consumer_id"] in test_consumers]

        split_info = {
            "strategy": strategy,
            "train_consumers": sorted(list(train_consumers)),
            "val_consumers": sorted(list(val_consumers)),
            "test_consumers": sorted(list(test_consumers)),
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "test_samples": len(test_indices),
        }

    elif strategy == "disjoint_pairs":
        unique_pairs = sorted(list(set(m["pair_id"] for m in metadata)))
        rng.shuffle(unique_pairs)

        n_p = len(unique_pairs)
        n_train = max(1, int(round(n_p * train_ratio)))
        n_val = max(1, int(round(n_p * val_ratio)))

        train_pairs = set(unique_pairs[:n_train])
        val_pairs = set(unique_pairs[n_train:n_train + n_val])
        test_pairs = set(unique_pairs[n_train + n_val:])

        if not test_pairs and len(val_pairs) > 1:
            test_pairs = {val_pairs.pop()}

        train_indices = [i for i, m in enumerate(metadata) if m["pair_id"] in train_pairs]
        val_indices = [i for i, m in enumerate(metadata) if m["pair_id"] in val_pairs]
        test_indices = [i for i, m in enumerate(metadata) if m["pair_id"] in test_pairs]

        split_info = {
            "strategy": strategy,
            "train_pairs": sorted(list(train_pairs)),
            "val_pairs": sorted(list(val_pairs)),
            "test_pairs": sorted(list(test_pairs)),
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "test_samples": len(test_indices),
        }

    elif strategy == "run_disjoint":
        # Partition unique run_ids into disjoint train/val/test groups.
        # Sequences from a given run_id are never split across groups.
        if not all("run_id" in m for m in metadata):
            raise ValueError("'run_disjoint' strategy requires 'run_id' field in all metadata entries.")

        from collections import defaultdict as _defaultdict
        run_to_indices: dict = _defaultdict(list)
        for i, m in enumerate(metadata):
            run_to_indices[m["run_id"]].append(i)

        all_run_ids = sorted(run_to_indices.keys())
        n_runs = len(all_run_ids)
        if n_runs < 3:
            raise ValueError(f"run_disjoint requires at least 3 runs; found {n_runs}.")

        rng.shuffle(all_run_ids)
        n_train_r = max(1, round(n_runs * train_ratio))
        n_val_r = max(1, round(n_runs * val_ratio))
        n_test_r = n_runs - n_train_r - n_val_r
        if n_test_r < 1:
            n_val_r -= 1
            n_test_r = 1

        train_runs_list = all_run_ids[:n_train_r]
        val_runs_list = all_run_ids[n_train_r: n_train_r + n_val_r]
        test_runs_list = all_run_ids[n_train_r + n_val_r:]

        train_indices = [i for rid in train_runs_list for i in run_to_indices[rid]]
        val_indices   = [i for rid in val_runs_list   for i in run_to_indices[rid]]
        test_indices  = [i for rid in test_runs_list  for i in run_to_indices[rid]]

        split_info = {
            "strategy": strategy,
            "n_runs_total": n_runs,
            "train_runs": sorted(train_runs_list),
            "val_runs": sorted(val_runs_list),
            "test_runs": sorted(test_runs_list),
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "test_samples": len(test_indices),
        }

    else:
        raise ValueError(f"Unknown split strategy '{strategy}'. "
                         f"Use 'vehicle_disjoint_consumer', 'disjoint_pairs', or 'run_disjoint'.")

    # Slice sets
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    meta_train = [metadata[i] for i in train_indices]
    meta_val = [metadata[i] for i in val_indices]
    meta_test = [metadata[i] for i in test_indices]

    return (
        (X_train, y_train, meta_train),
        (X_val, y_val, meta_val),
        (X_test, y_test, meta_test),
        split_info
    )
