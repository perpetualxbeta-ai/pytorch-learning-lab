"""Load a trained checkpoint and measure it on a fresh test set.

Usage:  python evaluate.py [--checkpoint checkpoints/tiny_mlp.pt] [--n-test 1000] [--seed 123]

This script depends only on the checkpoint file, never on anything left over
from a training run. If it works, saving and loading both work.
"""

import argparse

import torch

from dataset import make_circle_data
from model import TinyMLP


def load_model(path: str, device) -> TinyMLP:
    """Rebuild TinyMLP from a checkpoint and put it in eval mode on `device`."""
    # map_location lets a checkpoint saved on a GPU load on a CPU-only machine
    # (and vice versa). Without it, loading a CUDA checkpoint on a laptop fails.
    ckpt = torch.load(path, map_location=device)

    # Build an empty model with the SAME architecture, then copy the weights in.
    # PITFALL: if you rebuilt it by hand with a different size, e.g.
    # TinyMLP(hidden=32) for weights trained with hidden=16, load_state_dict
    # fails with "size mismatch for net.0.weight". That's why train.py saves
    # the config next to the weights.
    model = TinyMLP(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])

    # PITFALL: always switch to eval() before evaluating. This model gives the
    # same results either way, but a model with Dropout or BatchNorm would
    # give wrong (and randomly changing) results in train mode.
    return model.to(device).eval()


def predict(model, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (preds [N], probs [N, n_classes]) for inputs X [N, in_features]."""
    # PITFALL: no_grad() matters here too. Without it, autograd records a graph
    # for the whole test set that nothing will ever use, wasting memory.
    with torch.no_grad():
        logits = model(X)  # [N, 2] raw scores

    # The model outputs raw logits (no softmax, because CrossEntropyLoss adds
    # its own during training). Apply softmax only here, where we want readable
    # probabilities. dim=1 normalises across the classes, so each row sums to 1.
    probs = torch.softmax(logits, dim=1)  # [N, 2]

    # argmax of probs equals argmax of logits (softmax keeps the order), so
    # either gives the predicted class index for each sample.
    preds = probs.argmax(dim=1)  # [N]
    return preds, probs


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, n_classes: int = 2) -> torch.Tensor:
    """Count predictions per (true class, predicted class): returns [n_classes, n_classes]."""
    # Rows are the TRUE class, columns the PREDICTED class, so cm[t, p] counts
    # samples of class t that the model called p. The diagonal is correct.
    #
    # Vectorised trick: give each (t, p) pair its own index t * n_classes + p,
    # then count how often each index occurs. With 2 classes the indices are
    # 0 = (0,0), 1 = (0,1), 2 = (1,0), 3 = (1,1).
    # (The loop version, `for t, p in zip(...): cm[t, p] += 1`, does the same
    # thing much more slowly.)
    idx = targets * n_classes + preds  # [N]
    counts = torch.bincount(idx, minlength=n_classes * n_classes)  # [n_classes^2]
    return counts.view(n_classes, n_classes)  # [n_classes, n_classes]


def main() -> None:
    """Load the checkpoint, evaluate on fresh test data, and print a report."""
    p = argparse.ArgumentParser(description="Evaluate a trained TinyMLP.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/tiny_mlp.pt")
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)

    # A FRESH test set: seed 123 differs from the train (0) and val (1) seeds,
    # so the model has never seen these points.
    # PITFALL: reporting accuracy on the training data and calling it "test
    # accuracy". The model has fitted those exact points, so the number is
    # optimistic and tells you nothing about new data.
    # The test labels have no noise: we measure how well the model learned the
    # true circle, even if it was trained with --noise.
    X, y = make_circle_data(args.n_test, noise=0.0, seed=args.seed)
    X, y = X.to(device), y.to(device)  # same device as the model

    preds, probs = predict(model, X)  # [N], [N, 2]
    accuracy = (preds == y).float().mean().item()
    cm = confusion_matrix(preds, y, n_classes=probs.shape[1])

    print(f"test accuracy: {accuracy:.3f}  ({(preds == y).sum().item()}/{len(y)} correct)")
    print("\nconfusion matrix (rows = true, cols = predicted):")
    print("               pred outside  pred inside")
    print(f"true outside   {cm[0, 0].item():12d}  {cm[0, 1].item():11d}")
    print(f"true inside    {cm[1, 0].item():12d}  {cm[1, 1].item():11d}")

    print("\nsample predictions:")
    for i in range(5):
        x1, x2 = X[i].tolist()
        dist = (x1**2 + x2**2) ** 0.5  # inside if dist < 0.798
        print(
            f"  point ({x1:+.2f}, {x2:+.2f})  dist {dist:.2f}  true {y[i].item()}"
            f"  pred {preds[i].item()}  P(inside) {probs[i, 1].item():.3f}"
        )


if __name__ == "__main__":
    main()
