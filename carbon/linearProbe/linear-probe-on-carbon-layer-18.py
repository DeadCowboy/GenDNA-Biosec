# Single-layer linear probe run for the Carbon transformer, hooked via TransformerLens
# Flow: extract mean-pooled activations from layer 18 -> encode labels
#       -> train/test split -> train a linear probe
#       -> per-class accuracy graphic + training curves + checkpoint

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
MASKING_RATIO = 0
BATCH_SIZE = 30

SUBSET_SIZE = BATCH_SIZE * 100  # Number of examples in the subset
print(f"Sample size: {SUBSET_SIZE}")


# Params for Probe
layer = 18  # Layer to probe
hook = "hook_out"  # Hook to probe

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
GRAPHICS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "graphics-linear-probe-layer18")
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
# 1) Feature extraction: mean-pooled activations for every sequence
# ---------------------------------------------------------------------------
hook_name = f"blocks.{layer}.{hook}"

features_list = []  # one [batch, d_model] tensor per batch
labels_raw = []  # raw label strings, e.g. EC class

print("Extracting activations...")
for input_ids, attention_mask, label_dict in tqdm(dataloader, desc="Processing batches"):
    input_ids = input_ids.to(device)
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.to(device)

    _, cache = bridge.run_with_cache(input_ids, names_filter=hook_name, stop_at_layer=layer + 1)
    activations = cache[hook_name].float()  # [batch, pos, d_model]
    assert torch.isfinite(activations).all()  # If this fails something went badly wrong

    # Mean pooling along sequence dimension
    pooled = activations.mean(dim=1)  # [batch, d_model]
    features_list.append(pooled.cpu())

    # Labels
    for i in range(len(label_dict)):
        labels_raw.append(label_dict[i]["ec_number"][0])

    # Free GPU memory before the next batch
    del cache, activations, pooled
    if cuda_available:
        torch.cuda.empty_cache()

features = torch.cat(features_list, dim=0)  # [N, d_model]
features_list.clear()
labels_raw = np.array(labels_raw)
print(f"Collected {features.shape[0]} examples, d_model = {features.shape[1]}")


# ---------------------------------------------------------------------------
# 2) Label encoding (vocabulary over all collected labels)
# ---------------------------------------------------------------------------
classes = sorted(set(labels_raw.tolist()))
class_to_idx = {c: i for i, c in enumerate(classes)}
labels = torch.tensor([class_to_idx[l] for l in labels_raw], dtype=torch.long)
print(f"Classes ({len(classes)}): {classes}")


# ---------------------------------------------------------------------------
# 3) Train / test split (example-level)
# ---------------------------------------------------------------------------
# The subset was sampled without replacement, so each sequence appears once:
# an example-level split therefore has no sequence leakage. If your dataset
# can yield near-duplicates (same UniProt entry sampled twice), split on
# entry IDs instead.
n = features.shape[0]
n_train = int(TRAIN_FRACTION * n)
perm = torch.randperm(n)
train_idx, test_idx = perm[:n_train], perm[n_train:]

X_train, y_train = features[train_idx], labels[train_idx]
X_test, y_test = features[test_idx], labels[test_idx]
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

# Majority-class baseline, so the numbers below have context
majority = torch.bincount(y_train).argmax().item()
baseline_acc = (y_test == majority).float().mean().item()
print(f"Majority-class baseline: {baseline_acc:.4f}")


# ---------------------------------------------------------------------------
# 4) Standardise features (statistics computed on the train split only)
# ---------------------------------------------------------------------------
mean = X_train.mean(dim=0, keepdim=True)
std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# ---------------------------------------------------------------------------
# 5) Train the linear probe
# ---------------------------------------------------------------------------
probe = torch.nn.Linear(features.shape[1], len(classes)).to(device)
optimiser = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)

history = {"train_loss": [], "train_acc": [], "test_acc": []}

print("\nTraining linear probe...")
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
    print(f"Epoch {epoch + 1:3d} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | test_acc {test_acc:.4f}")


# ---------------------------------------------------------------------------
# 6) Final evaluation
# ---------------------------------------------------------------------------
probe.eval()
with torch.no_grad():
    preds = probe(X_test.to(device)).argmax(dim=-1).cpu()

test_acc = (preds == y_test).float().mean().item()
print(f"\nFinal test accuracy: {test_acc:.4f}")

# Per-class statistics on the test split
class_accuracy = []  # recall: correct predictions / true examples of the class
class_support = []  # number of test examples per class

print("\nPer-class accuracy / precision / recall / support:")
for c_idx, c in enumerate(classes):
    tp = ((preds == c_idx) & (y_test == c_idx)).sum().item()
    fp = ((preds == c_idx) & (y_test != c_idx)).sum().item()
    fn = ((preds != c_idx) & (y_test == c_idx)).sum().item()
    support = (y_test == c_idx).sum().item()

    accuracy = tp / max(support, 1)  # == recall
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    class_accuracy.append(accuracy)
    class_support.append(support)

    print(f"  {str(c):>10s}  acc {accuracy:.3f}  P {precision:.3f}  R {recall:.3f}  support {support}")


# ---------------------------------------------------------------------------
# 7) Graphic: per-class test accuracies
# ---------------------------------------------------------------------------
class_acc_path = os.path.join(GRAPHICS_DIR, f"layer{layer}_class_accuracies.png")

fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(classes)), 5))
x = np.arange(len(classes))
bars = ax.bar(x, class_accuracy, color="steelblue", edgecolor="black", linewidth=0.5)

# Annotate each bar with its accuracy and the class's test support
for xi, acc, support in zip(x, class_accuracy, class_support):
    ax.annotate(f"{acc:.3f}\n(n={support})", (xi, acc), textcoords="offset points", xytext=(0, 5), ha="center", va="bottom", fontsize=9)

ax.axhline(test_acc, color="red", linestyle="--", linewidth=1.2, label=f"overall test accuracy ({test_acc:.3f})")

ax.set_xticks(x)
ax.set_xticklabels([str(c) for c in classes])
ax.set_xlabel("EC class")
ax.set_ylabel("Test accuracy (recall)")
ax.set_ylim(0, 1.05)
ax.set_title(f"Per-class test accuracy — linear probe on layer {layer}")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(class_acc_path, dpi=150)
plt.close(fig)
print(f"Saved {class_acc_path}")


# ---------------------------------------------------------------------------
# 8) Training curves and checkpoint
# ---------------------------------------------------------------------------
curves_path = os.path.join(GRAPHICS_DIR, f"probe_layer{layer}_curves.png")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history["train_loss"])
axes[0].set_title(f"Train loss (layer {layer})")
axes[0].set_xlabel("Epoch")
axes[1].plot(history["train_acc"], label="train")
axes[1].plot(history["test_acc"], label="test")
axes[1].axhline(baseline_acc, color="grey", linestyle="--", linewidth=1, label="majority baseline")
axes[1].set_title(f"Accuracy (layer {layer})")
axes[1].set_xlabel("Epoch")
axes[1].legend()
fig.tight_layout()
fig.savefig(curves_path, dpi=150)
plt.close(fig)
print(f"Saved {curves_path}")

torch.save(
    {
        "state_dict": probe.state_dict(),
        "classes": classes,
        "feature_mean": mean,
        "feature_std": std,
        "layer": layer,
        "hook": hook,
        "repo": repo,
    },
    f"linear_probe_layer{layer}.pt",
)
print(f"Saved linear_probe_layer{layer}.pt")
