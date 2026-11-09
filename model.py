"""TinyMLP: a small fully connected network for 2-D, 2-class classification.

Only the network lives here. Training logic is in train.py, and loading and
prediction are in evaluate.py.

    input  [B, 2]
      Linear(2, 16)   -> [B, 16]
      ReLU
      Linear(16, 16)  -> [B, 16]
      ReLU
      Linear(16, 2)   -> [B, 2]   raw logits, one score per class
"""

import torch
from torch import nn


class TinyMLP(nn.Module):
    """Two hidden layers with ReLU, outputting raw class logits."""

    def __init__(self, in_features: int = 2, hidden: int = 16, n_classes: int = 2):
        # PITFALL: always call super().__init__() first. nn.Module sets up the
        # internal bookkeeping that registers sub-modules and parameters. If you
        # skip it, the next line fails with a confusing
        # "AttributeError: cannot assign module before Module.__init__() call".
        super().__init__()

        # Stored so evaluate.py (and the shape check below) know the sizes.
        self.in_features = in_features
        self.n_classes = n_classes

        # Assigning an nn.Module to an attribute (self.net = ...) registers it.
        # That's how model.parameters() finds every weight and bias, how the
        # optimizer knows what to update, and how .to(device) knows what to move.
        #
        # PITFALL: a plain Python list of layers, e.g.
        #     self.layers = [nn.Linear(2, 16), nn.Linear(16, 2)]
        # is NOT registered. Those weights never reach the optimizer, so they
        # are never trained, and .to(device) leaves them behind on the CPU.
        # Use nn.Sequential or nn.ModuleList instead.
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),  # [B, 2]  -> [B, 16]
            # The non-linearity is what lets the model bend its decision
            # boundary into a circle. Without the ReLUs, the three Linear layers
            # collapse into one linear map, which can only draw a straight line.
            nn.ReLU(),
            nn.Linear(hidden, hidden),  # [B, 16] -> [B, 16]
            nn.ReLU(),
            nn.Linear(hidden, n_classes),  # [B, 16] -> [B, 2]
            # PITFALL: no nn.Softmax here. CrossEntropyLoss applies log_softmax
            # itself, so adding Softmax would apply it twice. Training still
            # runs, but it learns slowly and poorly. evaluate.py applies softmax
            # only when it needs readable probabilities.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of points [B, in_features] to logits [B, n_classes]."""
        # nn.Linear works on the LAST dimension and treats everything before it
        # as batch dimensions. To predict a single point, give it a batch dim
        # first: x.unsqueeze(0) turns [2] into [1, 2]. (The assert below
        # rejects an unbatched [2] input.)
        logits = self.net(x)  # [B, n_classes]

        # Cheap shape check: catches a wrong input size right here instead of
        # as a confusing error later in the loss.
        assert logits.shape == (x.shape[0], self.n_classes), logits.shape
        return logits


def count_parameters(model: nn.Module) -> int:
    """Count the trainable parameters (those with requires_grad=True)."""
    # Each nn.Linear(in, out) holds a weight [out, in] and a bias [out], so it
    # has in*out + out parameters. numel() gives the element count of a tensor.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check only (build step 2 in SPEC.md): `python model.py`.
    model = TinyMLP()
    print(model)  # nn.Sequential prints every layer, which helps when debugging

    # PITFALL: call model(x), never model.forward(x). model(x) goes through
    # nn.Module.__call__, which runs PyTorch's hooks before and after forward.
    # Calling forward directly skips them; it works here but breaks tools
    # that rely on hooks.
    x = torch.randn(4, 2)  # a dummy batch of 4 points: [4, 2]
    logits = model(x)  # [4, 2]
    print(f"input {tuple(x.shape)} -> logits {tuple(logits.shape)}")
    print(f"trainable parameters: {count_parameters(model)}")
