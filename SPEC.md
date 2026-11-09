# SPEC: Tiny Neural Network on Synthetic Data

A small, modular PyTorch project for learning the basics. It trains a tiny
multi-layer perceptron (MLP) to classify 2-D points as **inside** or **outside**
a circle. The goal is understanding, not speed. Every file should be short
enough to read in one sitting, and every PyTorch idiom should be explained in
comments.

---

## 1. Goals and non-goals

**Goals**
- Show the full PyTorch workflow end to end: data → model → training loop → evaluation.
- Make tensor shapes visible at every step, in comments and in assertions.
- Point out common beginner mistakes where they tend to happen.
- Run on CPU in a few seconds. Use a GPU only if one is available.

**Non-goals**
- No real datasets, no downloads, no config frameworks, no logging libraries.
- No distributed training, mixed precision, or hyperparameter search.

---

## 2. The problem

Each sample is a point `x = (x1, x2)` drawn uniformly from the square `[-1, 1]²`.

```
label = 1  if  x1² + x2² < r²   (inside the circle)
label = 0  otherwise
```

The radius is `r = sqrt(2/π) ≈ 0.798`, so each class covers about half of the
square.

**Why this problem?**
- A straight line can't separate the classes. A model with no hidden layer
  (logistic regression) tops out around 50–60% accuracy, while a small MLP with
  a non-linearity can reach about 98%. This shows concretely why hidden layers
  and activations matter.
- The data is 2-D, so you can plot the points and the learned decision boundary
  (optional extension).
- An optional `noise` parameter flips a fraction of the labels, so the model
  can't reach 100%. This makes it possible to see overfitting.

---

## 3. Folder structure

```
pytorch-learning-lab/
├── SPEC.md            # this file
├── README.md          # how to install and run (short)
├── requirements.txt   # torch (and optionally matplotlib)
├── dataset.py         # synthetic data generation + Dataset/DataLoader helpers
├── model.py           # the nn.Module definition (TinyMLP)
├── train.py           # training loop; saves a checkpoint
├── evaluate.py        # loads a checkpoint; reports metrics on fresh test data
└── checkpoints/       # created at runtime by train.py (git-ignored)
```

Each file has one job. `train.py` and `evaluate.py` are the only entry points
(`python train.py`, `python evaluate.py`). `dataset.py` and `model.py` are
libraries that are imported, never run directly. This separation keeps each
file small and mirrors how larger PyTorch projects are laid out.

Dependencies only flow one way:

```
dataset.py   model.py
     ▲   ▲    ▲   ▲
     │   └────┼───┤
 train.py ────┘ evaluate.py
```

`train.py` and `evaluate.py` never import each other.

---

## 4. `dataset.py`

**Responsibility:** create the synthetic tensors and wrap them so a
`DataLoader` can batch and shuffle them.

### Public API

```python
def make_circle_data(n_samples: int, noise: float = 0.0, seed: int = 0
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (X, y).
    X: float32, shape [n_samples, 2]
    y: int64,   shape [n_samples]      # class indices 0 or 1
    """

class CircleDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples: int, noise: float = 0.0, seed: int = 0): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # returns (x_i, y_i) with shapes [2] and [] (a scalar tensor)

def get_dataloaders(n_train=2000, n_val=500, batch_size=64, noise=0.0, seed=0
                    ) -> tuple[DataLoader, DataLoader]:
    # train loader: shuffle=True; val loader: shuffle=False
```

### Design notes

- **Why subclass `Dataset`?** For a dataset this small, `TensorDataset(X, y)`
  would work. Writing `__len__` and `__getitem__` by hand shows the *protocol*
  that every `DataLoader` relies on, which you'll need later for images, text,
  and other data.
- **Why does `DataLoader` add the batch dimension?** `__getitem__` returns *one*
  sample with shape `[2]`. The `DataLoader` stacks `batch_size` of them to make
  `[B, 2]`. Keep that division of labor in mind.
- **Reproducibility:** use a local `torch.Generator().manual_seed(seed)` rather
  than the global seed, so train, val and test sets can use different seeds
  without overlapping.
- **Validation split** comes from a different seed, not from slicing the
  training tensor. This keeps the code simple and the sets independent.

### Pitfalls to comment in code
- Labels must be `torch.long` (int64) for `CrossEntropyLoss`. Float labels raise
  an error like *"expected scalar type Long but found Float"*.
- Labels must have shape `[N]`, not `[N, 1]`. A stray `unsqueeze` or `.view(-1, 1)`
  causes a shape error in the loss.
- Features must be `float32`. NumPy defaults to `float64`, and mixing it with the
  model's `float32` weights causes dtype errors.
- Shuffle the training loader, but not the validation loader.

---

## 5. `model.py`

**Responsibility:** define the network. No training logic lives here.

### Architecture

```
input  [B, 2]
  │  nn.Linear(2, 16)     →  [B, 16]
  │  nn.ReLU()
  │  nn.Linear(16, 16)    →  [B, 16]
  │  nn.ReLU()
  │  nn.Linear(16, 2)     →  [B, 2]    # raw logits, one per class
output [B, 2]
```

That's about 340 trainable parameters. `model.py` should include a
`count_parameters(model)` helper so you can check this number yourself.

### Public API

```python
class TinyMLP(nn.Module):
    def __init__(self, in_features: int = 2, hidden: int = 16, n_classes: int = 2): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_features] -> logits: [B, n_classes]

def count_parameters(model: nn.Module) -> int: ...
```

### Design notes

- **Why `nn.Module`?** Subclassing it registers every `nn.Linear` as a
  sub-module, so `model.parameters()` finds all weights automatically. That's
  how the optimizer knows what to update and how `.to(device)` knows what to move.
- **Why `nn.Sequential` inside `__init__`?** It keeps `forward` to a single line
  while still letting you print the model and see each layer. The spec also
  allows writing the layers out by hand in `forward`, which is more explicit,
  as long as comments mark the shapes.
- **Why 2 logits and `CrossEntropyLoss`, not 1 logit and `BCEWithLogitsLoss`?**
  The 2-logit setup extends to any number of classes without changes, and it
  avoids the `[B, 1]` vs `[B]` shape mismatch that commonly breaks BCE.
- **No softmax in `forward`.** `CrossEntropyLoss` applies `log_softmax`
  internally. Softmax is only applied in `evaluate.py` when probabilities are
  actually needed.

### Pitfalls to comment in code
- Forgetting `super().__init__()` means the layers are never registered, which
  leads to a confusing `AttributeError`.
- Calling `model.forward(x)` directly instead of `model(x)` skips PyTorch's
  hooks. Always use `model(x)`.
- Adding `nn.Softmax` before `CrossEntropyLoss` applies softmax twice. Training
  still runs, but it learns slowly and poorly.
- Putting layers in a plain Python `list` instead of `nn.ModuleList` or
  `nn.Sequential` leaves them unregistered, so they are never trained.

---

## 6. `train.py`

**Responsibility:** build the data and model, run the training loop, report
progress, and save a checkpoint.

### Flow

```
1. Parse a few CLI args (argparse): --epochs 20 --lr 1e-2 --batch-size 64
   --hidden 16 --noise 0.0 --seed 0 --out checkpoints/tiny_mlp.pt
2. torch.manual_seed(seed)                       # reproducible weight init
3. device = "cuda" if torch.cuda.is_available() else "cpu"
4. train_loader, val_loader = get_dataloaders(...)
5. model = TinyMLP(...).to(device)
6. loss_fn = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=lr)
7. for epoch in range(epochs):
       train_loss, train_acc = train_one_epoch(...)
       val_loss,   val_acc   = validate(...)
       print one line per epoch
8. torch.save({"model_state": model.state_dict(),
               "config": {...hyperparams...}}, out_path)
```

### Functions

```python
def train_one_epoch(model, loader, loss_fn, optimizer, device) -> tuple[float, float]
def validate(model, loader, loss_fn, device) -> tuple[float, float]
def main() -> None
```

### The core step (must appear with numbered comments)

```python
model.train()
for X, y in loader:
    X, y = X.to(device), y.to(device)   # (a) same device as the model
    optimizer.zero_grad()               # (b) clear gradients from the last step
    logits = model(X)                   # (c) forward:  [B, 2]
    loss = loss_fn(logits, y)           # (d) scalar;   y is [B] of int64
    loss.backward()                     # (e) autograd fills p.grad for every param
    optimizer.step()                    # (f) update params using p.grad
```

The comments must explain **autograd**. During the forward pass, PyTorch records
every operation on tensors that have `requires_grad=True`. `loss.backward()`
walks back through that record and *adds* gradients into each parameter's
`.grad`. Because it adds rather than replaces, you must call `zero_grad()` on
every step.

### Design notes
- **Why `Adam` at `lr=1e-2`?** It converges reliably on this problem in fewer
  than 20 epochs. SGD is a suggested experiment (see §9).
- **Why `validate()` uses `torch.no_grad()` and `model.eval()`:** `no_grad`
  stops autograd from recording, which saves memory and time. `eval()` switches
  layers like Dropout and BatchNorm to inference mode. This model has neither,
  but the habit matters.
- **Why save `state_dict()` and not the whole model?** A `state_dict` is just a
  dictionary of tensors. It's portable and doesn't depend on pickling your class
  definition. Saving the config alongside it lets `evaluate.py` rebuild the
  exact architecture.
- **Average the loss correctly:** accumulate `loss.item() * batch_size` and
  divide by the dataset size, because the last batch may be smaller.

### Pitfalls to comment in code
- Missing `optimizer.zero_grad()` makes gradients pile up across steps, so the
  loss blows up or behaves erratically.
- Device mismatch (*"Expected all tensors to be on the same device"*): the model
  and each batch both need `.to(device)`. Create the optimizer *after* calling
  `model.to(device)`.
- Accumulating `total_loss += loss` instead of `loss.item()` keeps the whole
  computation graph alive, so memory grows every step.
- Forgetting to switch back to `model.train()` after validation. Call it at the
  top of each `train_one_epoch`.
- Computing accuracy from `logits.argmax(dim=1)`: using the wrong `dim` silently
  gives garbage.

### Expected output (approximate)
```
epoch  1 | train loss 0.62 acc 0.66 | val loss 0.51 acc 0.79
...
epoch 20 | train loss 0.06 acc 0.98 | val loss 0.07 acc 0.98
saved checkpoint -> checkpoints/tiny_mlp.pt
```

---

## 7. `evaluate.py`

**Responsibility:** load a trained checkpoint and measure it on a **fresh test
set** (new seed, never seen in training or validation).

### Flow

```
1. CLI args: --checkpoint checkpoints/tiny_mlp.pt --n-test 1000 --seed 123
2. ckpt = torch.load(path, map_location=device)
3. model = TinyMLP(**ckpt["config"]); model.load_state_dict(ckpt["model_state"])
4. model.to(device).eval()
5. with torch.no_grad(): compute logits on the test set
6. Report: accuracy, 2×2 confusion matrix, and a few sample predictions
   with their softmax probabilities
```

### Functions

```python
def load_model(path: str, device) -> TinyMLP
def predict(model, X) -> tuple[torch.Tensor, torch.Tensor]  # (preds [N], probs [N, 2])
def confusion_matrix(preds, targets, n_classes=2) -> torch.Tensor  # [2, 2], pure torch
def main() -> None
```

### Design notes
- **Why a separate script?** Evaluation should depend only on a checkpoint file,
  not on variables left in memory from a training run. That also proves the
  save and load steps work.
- **Why `map_location`?** A checkpoint saved on a GPU can still be loaded on a
  CPU-only machine.
- The confusion matrix is computed with plain tensor ops (no scikit-learn) as
  indexing practice: `cm[t, p] += 1`, or a vectorized `torch.bincount`.
- Softmax is applied only here, with `torch.softmax(logits, dim=1)`, to turn
  logits into probabilities you can read.

### Pitfalls to comment in code
- Forgetting `model.eval()` and `torch.no_grad()`. Results are the same for this
  model but wrong for models with Dropout or BatchNorm, and memory use is higher.
- Rebuilding the model with a different `hidden` size than the one it was
  trained with makes `load_state_dict` fail with size-mismatch errors. That's
  why the config is saved in the checkpoint.
- Evaluating on the training data and calling it "test accuracy".

---

## 8. Coding conventions (apply to all files)

- Each file stays under about 150 lines. If one grows past that, split it.
- Every function that takes or returns a tensor has a shape comment, such as
  `# [B, 2]`.
- Key spots get a cheap runtime shape check, for example
  `assert logits.shape == (X.shape[0], n_classes)`.
- Mark beginner pitfalls with a consistent tag, `# PITFALL:`, so they're easy to
  find with grep.
- Use type hints on public functions and a one-line docstring on each.
- No global state. Pass everything as arguments.
- `.gitignore` includes `checkpoints/`, `__pycache__/` and `*.pt`.

---

## 9. Acceptance criteria

1. `pip install -r requirements.txt && python train.py` finishes in under 30 s on a laptop CPU.
2. With default settings, the final validation accuracy is at least 95%.
3. `python evaluate.py` loads the checkpoint and reports test accuracy within
   about 2 points of validation accuracy.
4. Running `train.py` twice with the same `--seed` gives identical losses on CPU.
5. `grep -rn "PITFALL" *.py` finds at least 3 entries in each of `dataset.py`,
   `model.py`, `train.py` and `evaluate.py`.

---

## 10. Suggested experiments (after it works)

| Experiment | What you'll learn |
|---|---|
| Remove both `ReLU`s | Without non-linearities, the stacked layers collapse into a single linear map and accuracy drops to about 50–60%. |
| Comment out `optimizer.zero_grad()` | What accumulating gradients does to the loss curve. |
| Switch to `torch.optim.SGD(lr=0.1)` | How much the choice of optimizer affects convergence speed. |
| `--noise 0.2 --hidden 128 --epochs 200` | Overfitting: train accuracy keeps rising while val accuracy stalls. |
| Change to 1 logit + `BCEWithLogitsLoss` | The `[B, 1]` vs `[B]` shape trap, and the `float` label requirement. |
| Plot the decision boundary (matplotlib) | What the network actually learned. |

---

## 11. Build order

Write and review the files one at a time, in this order:

1. `dataset.py`: generate data, print shapes and dtypes, check class balance is about 50/50.
2. `model.py`: pass a dummy `torch.randn(4, 2)` through the model and confirm the output is `[4, 2]`.
3. `train.py`: confirm the loss goes down.
4. `evaluate.py`: confirm test accuracy matches validation accuracy.
