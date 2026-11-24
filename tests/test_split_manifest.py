"""
Integrity tests for the run-disjoint large-dataset split (read-only).
Skipped automatically if the large dataset is not present.
"""
import json
import os
import unittest

import numpy as np

SPLIT_DIR = "data/splits/large_dataset"
META = "data/processed/large_dataset/metadata.json"
EXPECTED_TEST_RUNS = ["run_004", "run_005", "run_010", "run_012"]


@unittest.skipUnless(os.path.exists(META) and os.path.exists(f"{SPLIT_DIR}/split_info.json"),
                     "large dataset not available")
class TestSplitManifest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(f"{SPLIT_DIR}/split_info.json") as f:
            cls.info = json.load(f)
        with open(META) as f:
            cls.meta = json.load(f)
        cls.idx = {s: np.load(f"{SPLIT_DIR}/{s}_indices.npy") for s in ("train", "val", "test")}

    def runs_of(self, split):
        return sorted({self.meta[i]["run_id"] for i in self.idx[split]})

    def test_index_sets_disjoint(self):
        a, b, c = (set(self.idx[s].tolist()) for s in ("train", "val", "test"))
        self.assertFalse(a & b)
        self.assertFalse(a & c)
        self.assertFalse(b & c)
        self.assertEqual(len(a | b | c), len(self.meta))

    def test_recorded_runs_match_indices(self):
        for s in ("train", "val", "test"):
            self.assertEqual(self.runs_of(s), sorted(self.info[f"{s}_runs"]), s)

    def test_held_out_test_runs(self):
        self.assertEqual(self.runs_of("test"), EXPECTED_TEST_RUNS)

    def test_runs_disjoint(self):
        tr, va, te = (set(self.runs_of(s)) for s in ("train", "val", "test"))
        self.assertFalse(tr & va or tr & te or va & te)


if __name__ == "__main__":
    unittest.main()
