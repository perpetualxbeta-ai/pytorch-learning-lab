"""Synthetic "points inside a circle" data, wrapped for use with a DataLoader.

Each sample is a 2-D point drawn uniformly from the square [-1, 1]^2.
Label 1 means the point is inside a circle of radius sqrt(2/pi), label 0 means
outside. That radius makes the circle cover half the square, so the two classes
are balanced at about 50/50.
"""

import math

import torch
from torch.utils.data import DataLoader, Dataset

# Circle area = pi * r^2 = 2, which is half of the square's area (2 * 2 = 4).
RADIUS = math.sqrt(2 / math.pi)  # ~0.798


def make_circle_data(
    n_samples: int, noise: float = 0.0, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (X, y): X is float32 [n_samples, 2], y is int64 [n_samples]."""
    # A local Generator gives reproducible data without touching the global
    # RNG (torch.manual_seed). Different seeds give independent datasets, which
    # is how we keep train, val and test sets apart.
    gen = torch.Generator().manual_seed(seed)

    # torch.rand gives values in [0, 1). Scale and shift to get [-1, 1).
    # PITFALL: torch.rand already returns float32. If you build data with NumPy
    # instead, it defaults to float64, and feeding float64 into the model's
    # float32 weights raises "expected ... Float but found Double". Convert
    # with .float() if you ever go through NumPy.
    X = torch.rand(n_samples, 2, generator=gen) * 2 - 1  # [N, 2]

    # Squared distance from the origin, one value per point.
    # sum(dim=1) adds across the 2 columns (x1^2 + x2^2), so [N, 2] -> [N].
    # Using dim=0 would add down the rows and give shape [2], which is wrong.
    dist_sq = (X**2).sum(dim=1)  # [N]

    # PITFALL: CrossEntropyLoss needs class *indices* as int64 (torch.long).
    # The comparison gives a bool tensor, so convert it with .long().
    # Float labels fail with "expected scalar type Long but found Float".
    y = (dist_sq < RADIUS**2).long()  # [N]

    # Optional label noise: flip a random `noise` fraction of labels so the
    # task can't be solved perfectly (useful for seeing overfitting later).
    if noise > 0:
        flip = torch.rand(n_samples, generator=gen) < noise  # [N] bool mask
        y[flip] = 1 - y[flip]  # 0 -> 1 and 1 -> 0, only where flip is True

    # Cheap shape checks catch mistakes right where they happen.
    # PITFALL: y must be [N], not [N, 1]. A stray .unsqueeze(1) or
    # .view(-1, 1) here makes CrossEntropyLoss raise a shape error later.
    assert X.shape == (n_samples, 2) and X.dtype == torch.float32
    assert y.shape == (n_samples,) and y.dtype == torch.int64
    return X, y


class CircleDataset(Dataset):
    """Map-style Dataset: holds all tensors in memory, returns one sample at a time."""

    def __init__(self, n_samples: int, noise: float = 0.0, seed: int = 0):
        # Generate everything once up front. The data is tiny (a few KB), so
        # there's no reason to generate samples lazily in __getitem__.
        self.X, self.y = make_circle_data(n_samples, noise=noise, seed=seed)

    def __len__(self) -> int:
        # DataLoader calls this to know how many indices exist (0..len-1)
        # and how many batches make up one epoch.
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Return ONE sample: x_i has shape [2], y_i is a scalar tensor (shape []).
        # There is no batch dimension here. The DataLoader adds it by stacking
        # batch_size samples together: [2] -> [B, 2] and [] -> [B].
        # PITFALL: don't add a batch dim yourself (e.g. self.X[idx:idx+1]),
        # or the DataLoader gives you [B, 1, 2] batches.
        return self.X[idx], self.y[idx]


def get_dataloaders(
    n_train: int = 2000,
    n_val: int = 500,
    batch_size: int = 64,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build (train_loader, val_loader) from independently seeded datasets."""
    # The validation set uses a different seed, so its points are generated
    # separately from the training points instead of being sliced off them.
    # (evaluate.py uses yet another seed, 123 by default, for the test set.)
    # Label noise goes in both sets so val accuracy is measured on the same
    # task the model is trained on.
    train_ds = CircleDataset(n_train, noise=noise, seed=seed)
    val_ds = CircleDataset(n_val, noise=noise, seed=seed + 1)

    # PITFALL: shuffle the training data so each epoch sees batches in a new
    # order. Without it, the model sees the same sequence every epoch, which
    # can hurt training. Don't shuffle validation: order doesn't change the
    # metrics, and a fixed order makes runs easier to compare and debug.
    #
    # Passing our own seeded generator makes the shuffle order reproducible
    # without relying on the global RNG.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


if __name__ == "__main__":
    # Sanity check only (build step 1 in SPEC.md): `python dataset.py`.
    # The real entry points are train.py and evaluate.py.
    X, y = make_circle_data(2000)
    print(f"X: shape {tuple(X.shape)}, dtype {X.dtype}, range [{X.min():.2f}, {X.max():.2f}]")
    print(f"y: shape {tuple(y.shape)}, dtype {y.dtype}, fraction inside {y.float().mean():.3f}")

    train_loader, val_loader = get_dataloaders()
    xb, yb = next(iter(train_loader))
    print(f"one batch: X {tuple(xb.shape)}, y {tuple(yb.shape)}")
    print(f"batches per epoch: train {len(train_loader)}, val {len(val_loader)}")
