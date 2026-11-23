"""
PyTorch Dataset and DataLoader wrappers for Link-Survival sequences.
"""
from typing import Tuple, List, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LinkSurvivalDataset(Dataset):
    """PyTorch Dataset for Link Survival sequence tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray, metadata: List[Dict[str, Any]]):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_data_loaders(
    train_data: Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]],
    val_data: Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]],
    test_data: Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]],
    batch_size: int = 32,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds PyTorch DataLoaders for train, val, and test splits."""
    train_ds = LinkSurvivalDataset(*train_data)
    val_ds = LinkSurvivalDataset(*val_data)
    test_ds = LinkSurvivalDataset(*test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
