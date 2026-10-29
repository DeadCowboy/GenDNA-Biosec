# Linear-probe sweep over Carbon transformer layers, hooked via TransformerLens
# Flow: extract mean-pooled activations for layers 10..20 (stride 2) in one pass
#       -> encode labels -> shared train/test split
#       -> train a linear probe per layer -> save a training-curve figure per layer
#       -> save a summary figure of the best test accuracy per layer

from dataset.Enzyme.EnzymeDatasetAutoreg import SwissProtDataset

import sys
import os

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge

import numpy as np
import contextlib
import Levenshtein
from difflib import ndiff
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")  # headless: we save figures instead of showing them
import matplotlib.pyplot as plt

import inspect

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


SEQ_LENGTH = 500
MASKING_RATIO = 0.1
BATCH_SIZE = 30

SUBSET_SIZE = BATCH_SIZE * 100  # Number of examples in the subset
print(f"Sample size: {SUBSET_SIZE}")


# Params for Probe
hook = "hook_out"  # Hook to probe
PROBE_LAYERS = list(range(10, 26 + 1, 2))  # [10, 12, 14, 16, 18, 20]

# Linear-probe hyperparameters
TRAIN_FRACTION = 0.8  # fraction of examples used for training
PROBE_EPOCHS = 50
PROBE_LR = 1e-3
PROBE_BATCH_SIZE = 256
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

# Output folder: "graphics" in the parent directory of this script.
# It already exists, but makedirs(exist_ok=True) makes the script robust
# against being run from a checkout where it is missing.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHICS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "graphics2")
os.makedirs(GRAPHICS_DIR, exist_ok=True)
print(f"Graphics folder: {GRAPHICS_DIR}")


# Cuda
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

if cuda_available:
    print("GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Loading Model components
print("Initializing model and tokenizer...")
# repo = "HuggingFaceBio/Carbon-3B"
repo = "HuggingFaceBio/Carbon-500M"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

try:  # preflight: is the architecture supported by this TransformerLens?
    info = TransformerBridge.check_model_support(repo)
    print("Not supported") if not info["is_supported"] else print("Supported")
except Exception as e:  # informational only
    print(f"(check_model_support unavailable: {e})")

# Returns raw HF weights (no LayerNorm folding / centering) by default,
# which is what you want for probing the model's actual activations.
# Note: the bridge forces eager attention so hooks can fire, which is
# memory-hungry for very long sequences.
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
bridge = TransformerBridge.boot_transformers(
    repo,
    device=device,
    dtype=dtype,
    tokenizer=tok,
)
bridge.eval()


# Loading Dataset
dataset = SwissProtDataset(masking_ratio=MASKING_RATIO, seq_length=SEQ_LENGTH, length_multiple=6, ec_range="")

indices = torch.randperm(len(dataset))[:SUBSET_SIZE].tolist()
subset = Subset(dataset, indices)
dataloader = DataLoader(subset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn)


# ---------------------------------------------------------------------------
# 1) Feature extraction: mean-pooled activations for every probed layer
# ---------------------------------------------------------------------------
# All target hooks are captured in a single forward pass per batch
# (stop_at_layer just past the deepest layer), so the model runs once
# instead of once per layer. Caching several layers at once costs extra
# GPU memory: if you run out, lower BATCH_SIZE.
hook_names = [f"blocks.{L}.{hook}" for L in PROBE_LAYERS]  # names_filter accepts a list of hook names
stop_at_layer = max(PROBE_LAYERS) + 1

layer_features = {L: [] for L in PROBE_LAYERS}  # per layer: [batch, d_model] chunks
labels_raw = []  # raw label strings, e.g. EC class

print("Extracting activations...")
for input_ids, attention_mask, label_dict in tqdm(dataloader, desc="Processing batches"):
    input_ids = input_ids.to(device)
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.to(device)

    _, cache = bridge.run_with_cache(input_ids, names_filter=hook_names, stop_at_layer=stop_at_layer)

    for L in PROBE_LAYERS:
        hook_name = f"blocks.{L}.{hook}"
        activations = cache[hook_name].float()  # [batch, pos, d_model]
        assert torch.isfinite(activations).all()  # If this fails something went badly wrong

        # Mean pooling along sequence dimension
        pooled = activations.mean(dim=1)  # [batch, d_model]
        layer_features[L].append(pooled.cpu())
        del activations, pooled

    # Labels
    for i in range(len(label_dict)):
        labels_raw.append(label_dict[i]["ec_number"][0])

    # Free GPU memory before the next batch
    del cache
    if cuda_available:
        torch.cuda.empty_cache()

labels_raw = np.array(labels_raw)
n = len(labels_raw)
for L in PROBE_LAYERS:
    layer_features[L] = torch.cat(layer_features[L], dim=0)  # [N, d_model]
    assert layer_features[L].shape[0] == n
d_model = layer_features[PROBE_LAYERS[0]].shape[1]
print(f"Collected {n} examples, d_model = {d_model}, layers = {PROBE_LAYERS}")


# ---------------------------------------------------------------------------
# 2) Label encoding (vocabulary over all collected labels)
# ---------------------------------------------------------------------------
classes = sorted(set(labels_raw.tolist()))
class_to_idx = {c: i for i, c in enumerate(classes)}
labels = torch.tensor([class_to_idx[l] for l in labels_raw], dtype=torch.long)
print(f"Classes ({len(classes)}): {classes}")


# ---------------------------------------------------------------------------
# 3) Train / test split (example-level, shared across all layers)
# ---------------------------------------------------------------------------
# The subset was sampled without replacement, so each sequence appears once:
# an example-level split therefore has no sequence leakage. If your dataset
# can yield near-duplicates (same UniProt entry sampled twice), split on
# entry IDs instead. The same split is reused for every layer so the
# per-layer accuracies are directly comparable.
n_train = int(TRAIN_FRACTION * n)
perm = torch.randperm(n)
train_idx, test_idx = perm[:n_train], perm[n_train:]

y_train, y_test = labels[train_idx], labels[test_idx]
print(f"Train: {n_train}  Test: {n - n_train}")

# Majority-class baseline, so the numbers below have context
majority = torch.bincount(y_train).argmax().item()
baseline_acc = (y_test == majority).float().mean().item()
print(f"Majority-class baseline: {baseline_acc:.4f}")


# ---------------------------------------------------------------------------
# 4) Probe training helper
# ---------------------------------------------------------------------------
def standardise(X_train, X_test):
    """Standardise features; statistics come from the train split only."""
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def train_probe(X_train, y_train, X_test, y_test, num_classes):
    """Train a linear probe on standardised features.

    Returns the probe, its training history, and the best test accuracy
    seen during training (a simple early-stopping view of the run).
    """
    probe = torch.nn.Linear(X_train.shape[1], num_classes).to(device)
    optimiser = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    best_test_acc = 0.0

    for epoch in range(PROBE_EPOCHS):
        probe.train()
        order = torch.randperm(X_train.shape[0])
        total_loss, correct = 0.0, 0

        for start in range(0, X_train.shape[0], PROBE_BATCH_SIZE):
            batch = order[start : start + PROBE_BATCH_SIZE]
            xb = X_train[batch].to(device)
            yb = y_train[batch].to(device)

            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item() * xb.shape[0]
            correct += (logits.argmax(dim=-1) == yb).sum().item()

        train_loss = total_loss / X_train.shape[0]
        train_acc = correct / X_train.shape[0]

        probe.eval()
        with torch.no_grad():
            test_logits = probe(X_test.to(device))
            test_acc = (test_logits.argmax(dim=-1) == y_test.to(device)).float().mean().item()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        best_test_acc = max(best_test_acc, test_acc)

        print(f"  Epoch {epoch + 1:3d} | train_loss {train_loss:.4f} " f"| train_acc {train_acc:.4f} | test_acc {test_acc:.4f}")

    return probe, history, best_test_acc


def save_training_curves(history, layer, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title(f"Train loss (layer {layer})")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["test_acc"], label="test")
    axes[1].set_title(f"Accuracy (layer {layer})")
    axes[1].set_xlabel("Epoch")
    axes[1].axhline(baseline_acc, color="grey", linestyle="--", linewidth=1, label="majority baseline")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5) Sweep: train one probe per layer
# ---------------------------------------------------------------------------
results = {}  # layer -> (best_test_acc, probe, history, mean, std)

for L in PROBE_LAYERS:
    print(f"\n=== Layer {L} ===")
    features = layer_features[L]
    X_train, X_test = features[train_idx], features[test_idx]

    # Standardise features (statistics from the train split only)
    X_train, X_test, feat_mean, feat_std = standardise(X_train, X_test)

    probe, history, best_acc = train_probe(X_train, y_train, X_test, y_test, len(classes))
    results[L] = (best_acc, probe, history, feat_mean, feat_std)
    print(f"Layer {L}: best test accuracy {best_acc:.4f}")

    # Per-layer training curves
    curves_path = os.path.join(GRAPHICS_DIR, f"probe_layer{L}_curves.png")
    save_training_curves(history, L, curves_path)
    print(f"Saved {curves_path}")

    # Per-layer checkpoint
    torch.save(
        {
            "state_dict": probe.state_dict(),
            "classes": classes,
            "feature_mean": feat_mean,
            "feature_std": feat_std,
            "layer": L,
            "hook": hook,
            "repo": repo,
        },
        f"linear_probe_layer{L}.pt",
    )
    print(f"Saved linear_probe_layer{L}.pt")


# ---------------------------------------------------------------------------
# 6) Summary graphic: best test accuracy per layer
# ---------------------------------------------------------------------------
best_layer = max(results, key=lambda L: results[L][0])
summary_path = os.path.join(GRAPHICS_DIR, "accuracy_per_layer.png")

fig, ax = plt.subplots(figsize=(8, 5))
layers = PROBE_LAYERS
accs = [results[L][0] for L in layers]
ax.plot(layers, accs, marker="o", linewidth=2)
ax.axhline(baseline_acc, color="grey", linestyle="--", linewidth=1, label=f"majority baseline ({baseline_acc:.3f})")
for L, acc in zip(layers, accs):
    ax.annotate(f"{acc:.3f}", (L, acc), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax.set_xlabel("Layer")
ax.set_ylabel("Best test accuracy")
ax.set_title(f"Linear probe: best test accuracy per layer (best: layer {best_layer})")
ax.set_xticks(layers)
ax.legend()
fig.tight_layout()
fig.savefig(summary_path, dpi=150)
plt.close(fig)
print(f"\nSaved summary graphic {summary_path}")
print(f"Best layer: {best_layer} (best test accuracy {results[best_layer][0]:.4f})")


# ---------------------------------------------------------------------------
# 7) Per-class breakdown for the best layer
# ---------------------------------------------------------------------------
best_probe = results[best_layer][1]
X_best = layer_features[best_layer]
_, X_test_best, _, _ = standardise(X_best[train_idx], X_best[test_idx])
best_probe.eval()
with torch.no_grad():
    preds = best_probe(X_test_best.to(device)).argmax(dim=-1).cpu()

print("\nPer-class precision / recall / support (best layer):")
for c_idx, c in enumerate(classes):
    tp = ((preds == c_idx) & (y_test == c_idx)).sum().item()
    fp = ((preds == c_idx) & (y_test != c_idx)).sum().item()
    fn = ((preds != c_idx) & (y_test == c_idx)).sum().item()
    support = (y_test == c_idx).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"  {str(c):>10s}  P {precision:.3f}  R {recall:.3f}  support {support}")
