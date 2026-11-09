"""Train TinyMLP on the synthetic circle data and save a checkpoint.

Usage:  python train.py [--epochs 20] [--lr 1e-2] [--hidden 16] [--noise 0.0] ...
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from dataset import get_dataloaders
from model import TinyMLP, count_parameters


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> tuple[float, float]:
    """Run one pass over the training data; return (mean loss, accuracy)."""
    # PITFALL: validate() switches the model to eval mode. Switch back here at
    # the top of every epoch, or layers like Dropout/BatchNorm (if you add them
    # later) would silently stay in inference mode while training.
    model.train()
    total_loss, correct, n_seen = 0.0, 0, 0

    for X, y in loader:  # X: [B, 2] float32, y: [B] int64
        # (a) Move the batch to the same device as the model.
        # PITFALL: if the model is on "cuda" and the batch on "cpu", you get
        # "Expected all tensors to be on the same device". Both need .to(device).
        X, y = X.to(device), y.to(device)

        # (b) Clear the gradients left over from the previous step.
        # PITFALL: loss.backward() ADDS into each parameter's .grad, it doesn't
        # replace it. Forget zero_grad() and gradients pile up across steps,
        # so the updates get bigger and bigger and the loss behaves erratically.
        optimizer.zero_grad()

        # (c) Forward pass. While this runs, autograd records every operation
        # on tensors with requires_grad=True (all the weights and biases),
        # building a graph from the parameters to the loss.
        logits = model(X)  # [B, 2]

        # (d) Loss: one scalar for the whole batch (the mean over B samples).
        # CrossEntropyLoss wants logits [B, n_classes] and class indices [B].
        loss = loss_fn(logits, y)  # [] (scalar)

        # (e) Backward pass. Autograd walks the recorded graph from the loss
        # back to every parameter and adds d(loss)/d(param) into param.grad.
        # After this line, each p.grad has the same shape as p itself.
        loss.backward()

        # (f) Update: the optimizer reads each p.grad and adjusts p a little
        # in the direction that lowers the loss.
        optimizer.step()

        # Bookkeeping. The last batch can be smaller than batch_size
        # (2000 / 64 leaves 16), so weight each batch's mean loss by its size.
        # PITFALL: use loss.item() (a plain Python float). Writing
        # total_loss += loss keeps a reference to the whole graph of every
        # batch, so memory grows every step.
        b = y.shape[0]
        total_loss += loss.item() * b
        # argmax over dim=1 (the class dimension) picks the higher logit for
        # each sample: [B, 2] -> [B]. PITFALL: dim=0 would compare across the
        # batch instead and silently give a meaningless [2] tensor.
        correct += (logits.argmax(dim=1) == y).sum().item()
        n_seen += b

    return total_loss / n_seen, correct / n_seen


def validate(model, loader, loss_fn, device) -> tuple[float, float]:
    """Measure (mean loss, accuracy) on held-out data without updating weights."""
    # eval() switches Dropout/BatchNorm to inference mode. This model has
    # neither, but building the habit now saves you a nasty bug later.
    model.eval()
    total_loss, correct, n_seen = 0.0, 0, 0

    # no_grad() tells autograd not to record a graph: no backward pass is
    # coming, so recording would only waste memory and time.
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)  # [B, 2]
            b = y.shape[0]
            total_loss += loss_fn(logits, y).item() * b
            correct += (logits.argmax(dim=1) == y).sum().item()
            n_seen += b

    return total_loss / n_seen, correct / n_seen


def parse_args() -> argparse.Namespace:
    """Read hyperparameters from the command line."""
    p = argparse.ArgumentParser(description="Train TinyMLP on circle data.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints/tiny_mlp.pt")
    return p.parse_args()


def main() -> None:
    """Build data and model, train, and save a checkpoint."""
    args = parse_args()

    # Seeds the random weight initialisation, so the same --seed gives the
    # same starting weights and (on CPU) the same losses every run.
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size, noise=args.noise, seed=args.seed
    )

    model = TinyMLP(hidden=args.hidden).to(device)
    print(f"device: {device} | parameters: {count_parameters(model)}")

    loss_fn = nn.CrossEntropyLoss()
    # PITFALL: create the optimizer AFTER model.to(device). The optimizer holds
    # references to the parameter tensors, so build it from the final ones.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        print(
            f"epoch {epoch:2d} | train loss {train_loss:.3f} acc {train_acc:.3f}"
            f" | val loss {val_loss:.3f} acc {val_acc:.3f}"
        )

    # Save the weights (state_dict: a plain dict of name -> tensor) plus the
    # architecture settings, so evaluate.py can rebuild the exact same model
    # with TinyMLP(**config) before loading the weights into it.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config = {"in_features": model.in_features, "hidden": args.hidden, "n_classes": model.n_classes}
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "train_args": vars(args),  # for your records; not needed to load
        },
        out_path,
    )
    print(f"saved checkpoint -> {out_path}")


if __name__ == "__main__":
    main()
