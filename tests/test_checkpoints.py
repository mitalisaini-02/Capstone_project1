"""
Unit tests for checkpoint saving/loading (Phase 1: keyword-only save_checkpoint).
"""
import os
import tempfile
import unittest

import torch

from src.models.lstm import LinkSurvivalLSTM
from src.training.checkpoints import save_checkpoint, load_checkpoint


class TestCheckpoints(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.model = LinkSurvivalLSTM(input_dim=7, hidden_dim=16, num_layers=2, dropout=0.1)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "ckpt", "model.pt")

    def tearDown(self):
        self.tmp.cleanup()

    def test_keyword_round_trip(self):
        save_checkpoint(filepath=self.path, model=self.model, epoch=13, val_loss=0.33,
                        val_f1=0.78, best_threshold=0.53, config={"a": 1})
        fresh = LinkSurvivalLSTM(input_dim=7, hidden_dim=16, num_layers=2, dropout=0.1)
        ckpt = load_checkpoint(self.path, fresh)
        self.assertEqual(ckpt["epoch"], 13)
        self.assertAlmostEqual(ckpt["val_loss"], 0.33)
        self.assertAlmostEqual(ckpt["val_f1"], 0.78)
        self.assertAlmostEqual(ckpt["best_threshold"], 0.53)
        self.assertEqual(ckpt["config"], {"a": 1})
        for (k, v), (_, w) in zip(self.model.state_dict().items(), fresh.state_dict().items()):
            self.assertTrue(torch.equal(v, w), f"weight mismatch in {k}")

    def test_positional_call_is_rejected(self):
        # Reproduces the original bug: epoch passed positionally into the optimizer slot.
        with self.assertRaises(TypeError):
            save_checkpoint(self.path, self.model, 13, 0.33, 0.78, 0.53, {})
        self.assertFalse(os.path.exists(self.path))

    def test_optimizer_state_saved(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        save_checkpoint(filepath=self.path, model=self.model, optimizer=opt, epoch=1)
        ckpt = torch.load(self.path, weights_only=False)
        self.assertIsNotNone(ckpt["optimizer_state_dict"])


if __name__ == "__main__":
    unittest.main()
