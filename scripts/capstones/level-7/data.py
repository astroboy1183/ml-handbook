"""Level 7 capstone: FashionMNIST data pipeline.

Loads FashionMNIST once as uint8 tensors, splits the official training set into
train and validation with a fixed seed, and wraps each split in a Dataset that
applies torchvision.transforms v2 augmentation (training split only).
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class FashionDataset(Dataset):
    """Images as a uint8 tensor (N, 28, 28) plus labels; returns normalized float tensors."""

    def __init__(self, images, labels, mean, std, augment=False):
        self.images, self.labels = images, labels
        self.mean, self.std = mean, std
        # Light, label-preserving augmentation: shift by up to 2 pixels and mirror left-right.
        # Vertical flips or big rotations would create images that never occur at test time.
        self.augment = v2.Compose([v2.RandomCrop(28, padding=2), v2.RandomHorizontalFlip()]) if augment else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i].unsqueeze(0).float().div(255)      # (1, 28, 28) in [0, 1]
        if self.augment is not None:
            x = self.augment(x)
        return (x - self.mean) / self.std, self.labels[i]


def load_splits(data_dir="data", val_size=6000, seed=0):
    """Return (train, val, test) datasets. The test set is the official 10,000-image test split."""
    train_full = datasets.FashionMNIST(data_dir, train=True, download=True)
    test = datasets.FashionMNIST(data_dir, train=False, download=True)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_full.targets), generator=g)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    X, y = train_full.data, train_full.targets
    # Normalization statistics come from the training split only (no peeking at val or test).
    pixels = X[tr_idx].float().div(255)
    mean, std = pixels.mean().item(), pixels.std().item()
    train_ds = FashionDataset(X[tr_idx], y[tr_idx], mean, std, augment=True)
    val_ds = FashionDataset(X[val_idx], y[val_idx], mean, std)
    test_ds = FashionDataset(test.data, test.targets, mean, std)
    return train_ds, val_ds, test_ds


def make_loaders(train_ds, val_ds, test_ds, batch_size=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=512, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=512, num_workers=0)
    return train_dl, val_dl, test_dl
