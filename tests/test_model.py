"""
Unit tests for PyTorch LSTM architecture and tensor shapes.
"""
import unittest
import torch
from src.models.lstm import LinkSurvivalLSTM


class TestModel(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8
        self.seq_len = 10
        self.input_dim = 7
        self.model = LinkSurvivalLSTM(
            input_dim=self.input_dim,
            hidden_dim=32,
            num_layers=2,
            dropout=0.1
        )

    def test_forward_output_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        logits = self.model(x)
        self.assertEqual(logits.shape, (self.batch_size,))

    def test_predict_proba_bounds(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        probs = self.model.predict_proba(x)
        self.assertEqual(probs.shape, (self.batch_size,))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_gradient_flow(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        y = torch.tensor([0.0, 1.0] * 4)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        optimizer.zero_grad()
        logits = self.model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # Check gradients exist
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")


if __name__ == "__main__":
    unittest.main()
