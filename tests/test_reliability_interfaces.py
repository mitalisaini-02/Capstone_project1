"""Tests for the reliability interface, the LSTM adapter, the kinematic alternative and the threshold policy."""
import json
import os
import unittest

import numpy as np

from src.reliability.base import ReliabilityModel
from src.reliability.kinematic import KinematicSurvivalModel
from src.reliability.policy import ThresholdPolicy

CK = "outputs/large_dataset_experiment/checkpoints/"
META = CK + "checkpoint_metadata.json"
SENS = "outputs/large_dataset_experiment/reliability_threshold_robust/sensitivity_threshold.json"
FEATS = ["distance_m", "relative_speed_mps", "provider_speed_mps", "consumer_speed_mps",
         "delta_x_m", "delta_y_m", "closing_speed_mps"]


class TestThresholdPolicy(unittest.TestCase):
    def test_rule(self):
        p = ThresholdPolicy(0.62)
        self.assertTrue(p.is_reliable(0.62))
        self.assertFalse(p.is_reliable(0.6199))
        np.testing.assert_array_equal(p.mask([0.1, 0.62, 0.9]), [False, True, True])

    @unittest.skipUnless(os.path.exists(META) and os.path.exists(SENS), "threshold files missing")
    def test_recorded_thresholds_loaded_not_hardcoded(self):
        self.assertEqual(ThresholdPolicy.primary_from_metadata(META).theta, 0.62)
        self.assertEqual(ThresholdPolicy.sensitivity_from_file(SENS).theta, 0.65)

    def test_loader_validates_range(self):
        with self.assertRaises(ValueError):
            ThresholdPolicy(1.5).validated()


class _ConstantReliability(ReliabilityModel):
    def __init__(self, p):
        self.name, self.p = "constant", p
        self.label_range_m, self.history_window_s, self.horizon_s = 300.0, 10, 30
        self.feature_names = FEATS

    def predict_proba(self, windows):
        return np.full(len(windows), self.p)


class TestReplaceability(unittest.TestCase):
    def test_models_interchangeable_with_policy(self):
        w = np.random.default_rng(0).normal(size=(5, 10, 7)) * 100
        kin = KinematicSurvivalModel(1000.0, 30, 10, FEATS)
        for model in (kin, _ConstantReliability(0.7)):
            self.assertIsInstance(model, ReliabilityModel)
            p = model.predict_proba(w)
            self.assertEqual(p.shape, (5,))
            self.assertEqual(ThresholdPolicy(0.62).mask(p).shape, (5,))
        self.assertTrue(_ConstantReliability(0.7).range_mismatch(1000.0))
        self.assertFalse(kin.range_mismatch(1000.0))


@unittest.skipUnless(os.path.exists(CK + "best_model.pt") and os.path.exists(META), "checkpoint missing")
class TestLSTMAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.models.predictor import LinkSurvivalPredictor
        from src.reliability.lstm_distance import LSTMDistanceSurvivalModel
        cls.adapter = LSTMDistanceSurvivalModel(CK + "best_model.pt", CK + "model_config.json",
                                                CK + "scaler.json", META)
        cls.pred = LinkSurvivalPredictor(CK + "best_model.pt", CK + "model_config.json",
                                         CK + "scaler.json", device="cpu")
        cls.X = np.load("data/processed/large_dataset/pairwise_sequences.npz")["X"][:500].astype(np.float32)

    def test_declares_lstm_label_definition(self):
        a = self.adapter
        self.assertEqual((a.label_range_m, a.history_window_s, a.horizon_s), (1000.0, 10, 30))
        self.assertEqual(a.feature_names, FEATS)
        self.assertIn("not a radio link-quality guarantee", a.describe()["semantics"])

    def test_identical_to_predictor(self):
        np.testing.assert_array_equal(self.adapter.predict_proba(self.X),
                                      self.pred.predict_batch_probabilities(self.X))

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            self.adapter.predict_proba(self.X[:, :5, :])

    def test_predictor_threshold_backward_compatible(self):
        p = self.pred
        self.assertEqual(p.is_link_reliable(0.55), 0.55 >= p.tuned_threshold)
        self.assertTrue(p.is_link_reliable(0.62, threshold=0.62))
        self.assertFalse(p.is_link_reliable(0.99, threshold=1.5))   # out-of-range threshold still allowed


if __name__ == "__main__":
    unittest.main()
