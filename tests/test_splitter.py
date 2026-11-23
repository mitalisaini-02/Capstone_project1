"""
Unit tests for data splitting strategies and leakage prevention.
"""
import unittest
import numpy as np
from src.data.splitter import split_dataset


class TestSplitter(unittest.TestCase):

    def setUp(self):
        # 100 samples across 10 consumers (IDs 30 to 39) and 5 providers (IDs 0 to 4)
        N = 100
        self.X = np.random.randn(N, 10, 7)
        self.y = np.random.randint(0, 2, size=N)
        self.metadata = []
        for i in range(N):
            cid = 30 + (i % 10)
            pid = i % 5
            self.metadata.append({
                "time_s": 100.0 + i,
                "provider_id": pid,
                "consumer_id": cid,
                "pair_id": f"{pid}_{cid}",
                "label": int(self.y[i])
            })

    def test_vehicle_disjoint_consumer_split(self):
        train, val, test, info = split_dataset(
            self.X, self.y, self.metadata,
            strategy="vehicle_disjoint_consumer",
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            seed=42
        )
        _, _, meta_train = train
        _, _, meta_val = val
        _, _, meta_test = test

        train_c = set(m["consumer_id"] for m in meta_train)
        val_c = set(m["consumer_id"] for m in meta_val)
        test_c = set(m["consumer_id"] for m in meta_test)

        # STRICT ASSERTION: Zero consumer overlap across sets
        self.assertEqual(len(train_c & val_c), 0, "Leakage: Train and Val share consumers!")
        self.assertEqual(len(train_c & test_c), 0, "Leakage: Train and Test share consumers!")
        self.assertEqual(len(val_c & test_c), 0, "Leakage: Val and Test share consumers!")

    def test_pair_disjoint_split(self):
        train, val, test, info = split_dataset(
            self.X, self.y, self.metadata,
            strategy="disjoint_pairs",
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            seed=42
        )
        _, _, meta_train = train
        _, _, meta_val = val
        _, _, meta_test = test

        train_p = set(m["pair_id"] for m in meta_train)
        val_p = set(m["pair_id"] for m in meta_val)
        test_p = set(m["pair_id"] for m in meta_test)

        # STRICT ASSERTION: Zero pair overlap across sets
        self.assertEqual(len(train_p & val_p), 0, "Leakage: Train and Val share pairs!")
        self.assertEqual(len(train_p & test_p), 0, "Leakage: Train and Test share pairs!")
        self.assertEqual(len(val_p & test_p), 0, "Leakage: Val and Test share pairs!")


if __name__ == "__main__":
    unittest.main()
