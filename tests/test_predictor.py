"""
Unit tests for LinkSurvivalPredictor using a temporary checkpoint (no real artifacts touched).
"""
import json
import os
import tempfile
import unittest

import numpy as np
import torch

from src.features.scaler import SequenceStandardScaler
from src.models.lstm import LinkSurvivalLSTM
from src.models.predictor import LinkSurvivalPredictor
from src.training.checkpoints import save_checkpoint

CFG = {"name": "LinkSurvivalLSTM", "input_dim": 7, "hidden_dim": 16,
       "num_layers": 2, "dropout": 0.1, "bidirectional": False}


class TestPredictor(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.ckpt = os.path.join(d, "best_model.pt")
        self.cfg = os.path.join(d, "model_config.json")
        self.scaler = os.path.join(d, "scaler.json")
        self.meta = os.path.join(d, "checkpoint_metadata.json")

        model = LinkSurvivalLSTM(**{k: v for k, v in CFG.items() if k != "name"})
        save_checkpoint(filepath=self.ckpt, model=model, epoch=1, best_threshold=0.53)
        with open(self.cfg, "w") as f:
            json.dump(CFG, f)
        self.X = rng.normal(size=(20, 10, 7)).astype(np.float32) * 100
        SequenceStandardScaler(feature_names=[f"f{i}" for i in range(7)]).fit(self.X).save_json(self.scaler)
        with open(self.meta, "w") as f:
            json.dump({"reliability_threshold": {"value": 0.62}}, f)

    def tearDown(self):
        self.tmp.cleanup()

    def test_probabilities_in_unit_interval(self):
        p = LinkSurvivalPredictor(self.ckpt, self.cfg, self.scaler, device="cpu")
        probs = p.predict_batch_probabilities(self.X)
        self.assertEqual(probs.shape, (20,))
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))

    def test_single_matches_batch(self):
        p = LinkSurvivalPredictor(self.ckpt, self.cfg, self.scaler, device="cpu")
        batch = p.predict_batch_probabilities(self.X)
        single = np.array([p.predict_probability(self.X[i]) for i in range(len(self.X))])
        np.testing.assert_allclose(single, batch, atol=1e-6)

    def test_threshold_from_checkpoint_by_default(self):
        p = LinkSurvivalPredictor(self.ckpt, self.cfg, self.scaler, device="cpu")
        self.assertAlmostEqual(p.tuned_threshold, 0.53)
        self.assertEqual(p.threshold_source, "checkpoint")

    def test_threshold_from_metadata_overrides(self):
        p = LinkSurvivalPredictor(self.ckpt, self.cfg, self.scaler, device="cpu", metadata_path=self.meta)
        self.assertAlmostEqual(p.tuned_threshold, 0.62)
        self.assertEqual(p.threshold_source, "metadata")
        self.assertTrue(p.is_link_reliable(0.62))
        self.assertFalse(p.is_link_reliable(0.61))

    def test_missing_metadata_raises(self):
        with self.assertRaises(FileNotFoundError):
            LinkSurvivalPredictor(self.ckpt, self.cfg, self.scaler, device="cpu",
                                  metadata_path=os.path.join(self.tmp.name, "nope.json"))


if __name__ == "__main__":
    unittest.main()
