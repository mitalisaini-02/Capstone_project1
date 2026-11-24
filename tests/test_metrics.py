"""
Unit tests for evaluation metrics on small hand-computed cases.
"""
import unittest

import numpy as np

from src.evaluation.metrics import (compute_classification_metrics, compute_roc_auc,
                                    compute_pr_auc, tune_classification_threshold)


class TestMetrics(unittest.TestCase):

    def test_confusion_counts(self):
        y = np.array([1, 1, 0, 0, 1])
        p = np.array([0.9, 0.4, 0.6, 0.1, 0.8])
        m = compute_classification_metrics(y, p, threshold=0.5)
        self.assertEqual(m["confusion_matrix"], {"tp": 2, "fp": 1, "tn": 1, "fn": 1})
        self.assertAlmostEqual(m["precision"], round(2 / 3, 4))
        self.assertAlmostEqual(m["recall"], round(2 / 3, 4))
        self.assertAlmostEqual(m["specificity"], 0.5)

    def test_roc_auc_perfect_and_inverted(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(compute_roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertAlmostEqual(compute_roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)

    def test_roc_auc_hand_computed_with_tie(self):
        # pairs (pos,neg): (0.8,0.3)=1, (0.8,0.8)=0.5, (0.5,0.3)=1, (0.5,0.8)=0 -> 2.5/4
        y = np.array([1, 1, 0, 0])
        s = np.array([0.8, 0.5, 0.3, 0.8])
        self.assertAlmostEqual(compute_roc_auc(y, s), 0.625)

    def test_pr_auc_perfect(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(compute_pr_auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)

    def test_threshold_tuning_separable(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.2, 0.3, 0.4, 0.7, 0.75, 0.8])
        th, score = tune_classification_threshold(y, p, "balanced_accuracy")
        self.assertAlmostEqual(score, 1.0)
        self.assertTrue(0.4 < th <= 0.7)


if __name__ == "__main__":
    unittest.main()
