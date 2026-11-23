"""
split_large_dataset.py
-----------------------
Splits the combined large dataset using run-disjoint strategy.

Each SUMO run is treated as an atomic unit:
  - 70% of runs → training set
  - 15% of runs → validation set
  - 15% of runs → test set
  - NO sequence from a given run appears in more than one split

This eliminates leakage between splits that share vehicles or trajectories,
which would occur with sequence-level random splitting.

Saves:
  data/splits/large_dataset/split_info.json
  data/splits/large_dataset/train_indices.npy
  data/splits/large_dataset/val_indices.npy
  data/splits/large_dataset/test_indices.npy

Usage:
  python scripts/split_large_dataset.py
      [--dataset_dir data/processed/large_dataset]
      [--out_dir data/splits/large_dataset]
      [--train_ratio 0.70] [--val_ratio 0.15] [--seed 42]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_disjoint_split(
    metadata: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Partition run_ids into train/val/test groups (whole runs),
    then assign sequence indices accordingly.

    Returns split_info dict with index arrays.
    """
    # Group sequence indices by run_id
    run_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        run_id = m.get("run_id", "unknown")
        run_to_indices[run_id].append(i)

    all_run_ids = sorted(run_to_indices.keys())
    n_runs = len(all_run_ids)

    if n_runs < 3:
        raise ValueError(
            f"Need at least 3 runs for a 3-way split, but only {n_runs} runs found."
        )

    rng = random.Random(seed)
    shuffled = all_run_ids.copy()
    rng.shuffle(shuffled)

    n_train = max(1, round(n_runs * train_ratio))
    n_val = max(1, round(n_runs * val_ratio))
    # Ensure at least 1 run in test
    n_test = n_runs - n_train - n_val
    if n_test < 1:
        # Steal from val
        n_val -= 1
        n_test = 1

    train_runs = set(shuffled[:n_train])
    val_runs = set(shuffled[n_train: n_train + n_val])
    test_runs = set(shuffled[n_train + n_val:])

    # Collect indices
    train_idx = []
    val_idx = []
    test_idx = []
    for rid in shuffled[:n_train]:
        train_idx.extend(run_to_indices[rid])
    for rid in shuffled[n_train: n_train + n_val]:
        val_idx.extend(run_to_indices[rid])
    for rid in shuffled[n_train + n_val:]:
        test_idx.extend(run_to_indices[rid])

    # Verify disjoint
    s_train = set(train_idx)
    s_val = set(val_idx)
    s_test = set(test_idx)
    assert s_train.isdisjoint(s_val), "Train/Val overlap detected!"
    assert s_train.isdisjoint(s_test), "Train/Test overlap detected!"
    assert s_val.isdisjoint(s_test), "Val/Test overlap detected!"

    return {
        "strategy": "run_disjoint",
        "seed": seed,
        "n_runs_total": n_runs,
        "n_runs_train": len(train_runs),
        "n_runs_val": len(val_runs),
        "n_runs_test": len(test_runs),
        "train_runs": sorted(list(train_runs)),
        "val_runs": sorted(list(val_runs)),
        "test_runs": sorted(list(test_runs)),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run-disjoint split for large dataset.")
    parser.add_argument("--dataset_dir", type=str, default="data/processed/large_dataset",
                        help="Directory containing pairwise_sequences.npz and metadata.json.")
    parser.add_argument("--out_dir", type=str, default="data/splits/large_dataset",
                        help="Output directory for split info and index arrays.")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / args.dataset_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: metadata.json not found at {meta_path}")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    # Load arrays to get label distribution per split
    npz_path = dataset_dir / "pairwise_sequences.npz"
    if not npz_path.exists():
        print(f"ERROR: pairwise_sequences.npz not found at {npz_path}")
        sys.exit(1)

    data = np.load(str(npz_path))
    y = data["y"]

    print(f"Dataset: {len(metadata):,} sequences | {npz_path}")
    print(f"Labels: pos={int(np.sum(y==1)):,} neg={int(np.sum(y==0)):,}")

    # Split
    split_info = run_disjoint_split(
        metadata=metadata,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Annotate with label distributions
    for split_name, key in [("train", "train_indices"), ("val", "val_indices"), ("test", "test_indices")]:
        idx = split_info[key]
        y_split = y[idx]
        split_info[f"{split_name}_pos"] = int(np.sum(y_split == 1))
        split_info[f"{split_name}_neg"] = int(np.sum(y_split == 0))
        pos_frac = split_info[f"{split_name}_pos"] / max(len(idx), 1)
        split_info[f"{split_name}_pos_fraction"] = round(pos_frac, 4)

    # Verify: no run appears in multiple splits
    train_runs = set(split_info["train_runs"])
    val_runs = set(split_info["val_runs"])
    test_runs = set(split_info["test_runs"])
    assert train_runs.isdisjoint(val_runs), "Run leakage: train ∩ val"
    assert train_runs.isdisjoint(test_runs), "Run leakage: train ∩ test"
    assert val_runs.isdisjoint(test_runs), "Run leakage: val ∩ test"
    print("  ✓ Run-disjoint split verified — zero run overlap across splits.")

    # Save index arrays
    train_idx = np.array(split_info["train_indices"], dtype=np.int64)
    val_idx = np.array(split_info["val_indices"], dtype=np.int64)
    test_idx = np.array(split_info["test_indices"], dtype=np.int64)
    np.save(str(out_dir / "train_indices.npy"), train_idx)
    np.save(str(out_dir / "val_indices.npy"), val_idx)
    np.save(str(out_dir / "test_indices.npy"), test_idx)

    # Save split_info (without index arrays for readability)
    info_to_save = {k: v for k, v in split_info.items()
                    if k not in ("train_indices", "val_indices", "test_indices")}
    split_info_path = out_dir / "split_info.json"
    split_info_path.write_text(json.dumps(info_to_save, indent=2))

    print(f"\nSPLIT SUMMARY (run-disjoint)")
    print(f"  Runs:    train={split_info['n_runs_train']}  val={split_info['n_runs_val']}  test={split_info['n_runs_test']}")
    print(f"  Samples: train={split_info['train_samples']:,}  val={split_info['val_samples']:,}  test={split_info['test_samples']:,}")
    print(f"  Pos frac: train={split_info['train_pos_fraction']:.3f}  val={split_info['val_pos_fraction']:.3f}  test={split_info['test_pos_fraction']:.3f}")
    print(f"\n  Saved:")
    print(f"    {out_dir}/split_info.json")
    print(f"    {out_dir}/train_indices.npy  ({len(train_idx):,} indices)")
    print(f"    {out_dir}/val_indices.npy    ({len(val_idx):,} indices)")
    print(f"    {out_dir}/test_indices.npy   ({len(test_idx):,} indices)")


if __name__ == "__main__":
    main()
