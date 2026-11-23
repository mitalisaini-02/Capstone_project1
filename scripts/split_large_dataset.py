"""
split_large_dataset.py
-----------------------
Splits the combined large dataset using strict run-disjoint strategy.

Each SUMO simulation run is an atomic unit:
  - ~70% of runs → training set
  - ~15% of runs → validation set
  - ~15% of runs → test set

Guarantees:
  - Zero run overlap
  - Zero vehicle overlap (vehicle IDs prefixed by run_id)
  - Zero pair overlap

Saves:
  data/splits/large_dataset/split_info.json
  data/splits/large_dataset/split_audit.json
  data/splits/large_dataset/train_indices.npy
  data/splits/large_dataset/val_indices.npy
  data/splits/large_dataset/test_indices.npy
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_disjoint_split(
    metadata: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    run_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(metadata):
        run_id = m.get("run_id", "unknown")
        run_to_indices[run_id].append(i)

    all_run_ids = sorted(run_to_indices.keys())
    n_runs = len(all_run_ids)

    if n_runs < 3:
        raise ValueError(f"Need at least 3 runs for a 3-way split, but only {n_runs} runs found.")

    # Find optimal combination of runs to match target ratios closely while preserving atomic run partition
    import itertools
    total_seqs = len(metadata)
    target_tr = train_ratio * total_seqs
    target_va = val_ratio * total_seqs
    target_te = (1.0 - train_ratio - val_ratio) * total_seqs

    best_diff = float("inf")
    best_train_runs = None
    best_val_runs = None
    best_test_runs = None

    # Search over candidate set sizes for val and test
    for k_val in range(2, max(3, n_runs // 3)):
        for val_candidate in itertools.combinations(all_run_ids, k_val):
            val_sum = sum(len(run_to_indices[r]) for r in val_candidate)
            rem = [r for r in all_run_ids if r not in val_candidate]
            for k_test in range(2, max(3, n_runs // 3)):
                for test_candidate in itertools.combinations(rem, k_test):
                    test_sum = sum(len(run_to_indices[r]) for r in test_candidate)
                    train_candidate = [r for r in rem if r not in test_candidate]
                    train_sum = sum(len(run_to_indices[r]) for r in train_candidate)

                    diff = abs(train_sum - target_tr) + abs(val_sum - target_va) + abs(test_sum - target_te)
                    if diff < best_diff:
                        best_diff = diff
                        best_train_runs = train_candidate
                        best_val_runs = list(val_candidate)
                        best_test_runs = list(test_candidate)

    train_runs = set(best_train_runs)
    val_runs = set(best_val_runs)
    test_runs = set(best_test_runs)

    train_idx = [i for rid in train_runs for i in run_to_indices[rid]]
    val_idx = [i for rid in val_runs for i in run_to_indices[rid]]
    test_idx = [i for rid in test_runs for i in run_to_indices[rid]]

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


def audit_splits(metadata: List[Dict[str, Any]], split_info: Dict[str, Any]) -> Dict[str, Any]:
    """Audits run, vehicle, and pair disjointness across splits."""
    train_runs = set(split_info["train_runs"])
    val_runs = set(split_info["val_runs"])
    test_runs = set(split_info["test_runs"])

    train_idx = set(split_info["train_indices"])
    val_idx = set(split_info["val_indices"])
    test_idx = set(split_info["test_indices"])

    train_vehicles: Set[str] = set()
    val_vehicles: Set[str] = set()
    test_vehicles: Set[str] = set()

    train_pairs: Set[str] = set()
    val_pairs: Set[str] = set()
    test_pairs: Set[str] = set()

    for i, m in enumerate(metadata):
        pid = str(m.get("provider_id"))
        cid = str(m.get("consumer_id"))
        pair_id = str(m.get("pair_id"))
        rid = str(m.get("run_id"))

        v1 = f"{rid}__{pid}" if "__" not in pid else pid
        v2 = f"{rid}__{cid}" if "__" not in cid else cid

        if i in train_idx:
            train_vehicles.add(v1)
            train_vehicles.add(v2)
            train_pairs.add(pair_id)
        elif i in val_idx:
            val_vehicles.add(v1)
            val_vehicles.add(v2)
            val_pairs.add(pair_id)
        elif i in test_idx:
            test_vehicles.add(v1)
            test_vehicles.add(v2)
            test_pairs.add(pair_id)

    # Compute overlaps
    run_overlap_tr_val = list(train_runs & val_runs)
    run_overlap_tr_te = list(train_runs & test_runs)
    run_overlap_val_te = list(val_runs & test_runs)

    veh_overlap_tr_val = list(train_vehicles & val_vehicles)
    veh_overlap_tr_te = list(train_vehicles & test_vehicles)
    veh_overlap_val_te = list(val_vehicles & test_vehicles)

    pair_overlap_tr_val = list(train_pairs & val_pairs)
    pair_overlap_tr_te = list(train_pairs & test_pairs)
    pair_overlap_val_te = list(val_pairs & test_pairs)

    passed = (
        len(run_overlap_tr_val) == 0 and len(run_overlap_tr_te) == 0 and len(run_overlap_val_te) == 0 and
        len(veh_overlap_tr_val) == 0 and len(veh_overlap_tr_te) == 0 and len(veh_overlap_val_te) == 0 and
        len(pair_overlap_tr_val) == 0 and len(pair_overlap_tr_te) == 0 and len(pair_overlap_val_te) == 0
    )

    return {
        "audit_passed": passed,
        "runs": {
            "train": sorted(list(train_runs)),
            "val": sorted(list(val_runs)),
            "test": sorted(list(test_runs)),
            "overlap_train_val": run_overlap_tr_val,
            "overlap_train_test": run_overlap_tr_te,
            "overlap_val_test": run_overlap_val_te,
        },
        "vehicles": {
            "train_count": len(train_vehicles),
            "val_count": len(val_vehicles),
            "test_count": len(test_vehicles),
            "overlap_train_val_count": len(veh_overlap_tr_val),
            "overlap_train_test_count": len(veh_overlap_tr_te),
            "overlap_val_test_count": len(veh_overlap_val_te),
        },
        "pairs": {
            "train_count": len(train_pairs),
            "val_count": len(val_pairs),
            "test_count": len(test_pairs),
            "overlap_train_val_count": len(pair_overlap_tr_val),
            "overlap_train_test_count": len(pair_overlap_tr_te),
            "overlap_val_test_count": len(pair_overlap_val_te),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run-disjoint split for large dataset.")
    parser.add_argument("--dataset_dir", type=str, default="data/processed/large_dataset")
    parser.add_argument("--out_dir", type=str, default="data/splits/large_dataset")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / args.dataset_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = dataset_dir / "metadata.json"
    npz_path = dataset_dir / "pairwise_sequences.npz"

    with open(meta_path) as f:
        metadata = json.load(f)

    data = np.load(str(npz_path))
    y = data["y"]

    print("=" * 70)
    print(" SPLITTING DATASET (RUN-DISJOINT)")
    print(f" Total sequences : {len(metadata):,}")
    print(f" Labels          : pos={int(np.sum(y==1)):,} neg={int(np.sum(y==0)):,}")
    print("=" * 70)

    split_info = run_disjoint_split(
        metadata=metadata,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    for split_name, key in [("train", "train_indices"), ("val", "val_indices"), ("test", "test_indices")]:
        idx = split_info[key]
        y_split = y[idx]
        split_info[f"{split_name}_pos"] = int(np.sum(y_split == 1))
        split_info[f"{split_name}_neg"] = int(np.sum(y_split == 0))
        pos_frac = split_info[f"{split_name}_pos"] / max(len(idx), 1)
        split_info[f"{split_name}_pos_fraction"] = round(pos_frac, 4)

    audit_result = audit_splits(metadata, split_info)
    assert audit_result["audit_passed"], f"Audit failed: {audit_result}"

    print(f"\n  ✓ Audit Passed: ZERO run overlap, ZERO vehicle overlap, ZERO pair overlap.")

    # Save indices
    train_idx = np.array(split_info["train_indices"], dtype=np.int64)
    val_idx = np.array(split_info["val_indices"], dtype=np.int64)
    test_idx = np.array(split_info["test_indices"], dtype=np.int64)
    np.save(str(out_dir / "train_indices.npy"), train_idx)
    np.save(str(out_dir / "val_indices.npy"), val_idx)
    np.save(str(out_dir / "test_indices.npy"), test_idx)

    info_to_save = {k: v for k, v in split_info.items() if k not in ("train_indices", "val_indices", "test_indices")}
    (out_dir / "split_info.json").write_text(json.dumps(info_to_save, indent=2))
    (out_dir / "split_audit.json").write_text(json.dumps(audit_result, indent=2))

    total = len(train_idx) + len(val_idx) + len(test_idx)
    print("\n" + "=" * 70)
    print(" SPLIT SIZES & PERCENTAGES")
    print("=" * 70)
    print(f" Train : {len(train_idx):,} sequences ({100*len(train_idx)/total:.1f}%) | Runs: {split_info['train_runs']}")
    print(f" Val   : {len(val_idx):,} sequences ({100*len(val_idx)/total:.1f}%) | Runs: {split_info['val_runs']}")
    print(f" Test  : {len(test_idx):,} sequences ({100*len(test_idx)/total:.1f}%) | Runs: {split_info['test_runs']}")
    print(f"\n Saved split metadata to {out_dir}")


if __name__ == "__main__":
    main()
