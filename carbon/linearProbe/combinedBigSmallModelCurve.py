# Linear-probe sweep over Carbon transformer layers, hooked via TransformerLens
# Flow: for each model (Carbon-3B and Carbon-500M):
#       extract mean-pooled activations for the probed layers in one pass
#       -> encode labels -> shared train/test split (identical for both models)
#       -> train a linear probe per layer
#       -> finally save one summary figure comparing both models' accuracies

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
BATCH_SIZE = 20

SUBSET_SIZE = BATCH_SIZE * 500  # Number of examples in the subset
print(f"Sample size: {SUBSET_SIZE}")


# Params for Probe
hook = "hook_out"  # Hook to probe
DESIRED_PROBE_LAYERS = list(range(8, 28 + 1, 2))  # clamped per model to its depth

# Models to sweep. Both share the same labels, split and probe
# hyperparameters, so their per-layer accuracies are directly comparable.
MODELS = [
    {"name": "Carbon-3B", "repo": "HuggingFaceBio/Carbon-3B", "color": "tab:blue"},
    {"name": "Carbon-500M", "repo": "HuggingFaceBio/Carbon-500M", "color": "tab:orange"},
]

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
GRAPHICS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "graphics-BigRun")
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


# Loading Dataset
dataset = SwissProtDataset(masking_ratio=MASKING_RATIO, seq_length=SEQ_LENGTH, length_multiple=6, ec_range="")

indices = torch.randperm(len(dataset))[:SUBSET_SIZE].tolist()
subset = Subset(dataset, indices)
dataloader = DataLoader(subset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_num_layers(bridge):
    """Best-effort extraction of the model's layer count (TL or HF style)."""
    candidates = [
        lambda: bridge.cfg.n_layers,
        lambda: bridge.tl_model.cfg.n_layers,
        lambda: bridge.model.cfg.n_layers,
        lambda: bridge.hf_model.config.num_hidden_layers,
        lambda: bridge.model.config.num_hidden_layers,
        lambda: bridge.config.num_hidden_layers,
    ]
    for fn in candidates:
        try:
            value = int(fn())
            if value > 0:
                return value
        except Exception:
            continue
    raise RuntimeError("Could not determine the model's layer count; "
                       "set the probe layers manually for this model.")


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

        print(f"  Epoch {epoch + 1:3d} | train_loss {train_loss:.4f} "
              f"| train_acc {train_acc:.4f} | test_acc {test_acc:.4f}")

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
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def extract_features(bridge, dataloader, probe_layers):
    """Mean-pooled activations for every probed layer, captured in a single
    forward pass per batch (stop_at_layer just past the deepest layer), so
    the model runs once instead of once per layer. Caching several layers at
    once costs extra GPU memory: if you run out, lower BATCH_SIZE."""
    hook_names = [f"blocks.{L}.{hook}" for L in probe_layers]  # names_filter accepts a list
    stop_at_layer = max(probe_layers) + 1

    feats = {L: [] for L in probe_layers}  # per layer: [batch, d_model] chunks
    labels_raw = []  # raw label strings, e.g. EC class

    print("Extracting activations...")
    for input_ids, attention_mask, label_dict in tqdm(dataloader, desc="Processing batches"):
        input_ids = input_ids.to(device)
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(device)

        _, cache = bridge.run_with_cache(input_ids, names_filter=hook_names, stop_at_layer=stop_at_layer)

        for L in probe_layers:
            hook_name = f"blocks.{L}.{hook}"
            activations = cache[hook_name].float()  # [batch, pos, d_model]
            assert torch.isfinite(activations).all()  # If this fails something went badly wrong

            # Mean pooling along sequence dimension
            pooled = activations.mean(dim=1)  # [batch, d_model]
            feats[L].append(pooled.cpu())
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
    for L in probe_layers:
        feats[L] = torch.cat(feats[L], dim=0)  # [N, d_model]
        assert feats[L].shape[0] == n
    print(f"Collected {n} examples, d_model = {feats[probe_layers[0]].shape[1]}, layers = {probe_layers}")
    return feats, labels_raw


def per_class_breakdown(model_name, best_layer, probe, features, train_idx, test_idx, classes, y_test):
    _, X_test, _, _ = standardise(features[train_idx], features[test_idx])
    probe.eval()
    with torch.no_grad():
        preds = probe(X_test.to(device)).argmax(dim=-1).cpu()

    print(f"\nPer-class precision / recall / support ({model_name}, best layer {best_layer}):")
    for c_idx, c in enumerate(classes):
        tp = ((preds == c_idx) & (y_test == c_idx)).sum().item()
        fp = ((preds == c_idx) & (y_test != c_idx)).sum().item()
        fn = ((preds != c_idx) & (y_test == c_idx)).sum().item()
        support = (y_test == c_idx).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        print(f"  {str(c):>10s}  P {precision:.3f}  R {recall:.3f}  support {support}")


# ---------------------------------------------------------------------------
# Shared state: labels, classes and the train/test split are derived from the
# FIRST extraction pass and reused verbatim for every model, so per-layer
# accuracies are directly comparable across models.
# ---------------------------------------------------------------------------
labels_raw = None
classes = None
class_to_idx = None
labels = None
train_idx = test_idx = None
y_train = y_test = None

all_results = {}         # model name -> {layer: (best_acc, probe, history, mean, std)}
model_probe_layers = {}  # model name -> list of probed layers
model_best_layer = {}     # model name -> best layer


for model_idx, model in enumerate(MODELS):
    name, repo = model["name"], model["repo"]
    print(f"\n{'=' * 70}\n=== Model {model_idx + 1}/{len(MODELS)}: {name} ({repo})\n{'=' * 70}")

    # Loading Model components
    print("Initializing model and tokenizer...")
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

    # Clamp the desired sweep to the layers this model actually has
    n_layers = get_num_layers(bridge)
    probe_layers = [L for L in DESIRED_PROBE_LAYERS if L < n_layers]
    if len(probe_layers) < len(DESIRED_PROBE_LAYERS):
        print(f"Note: {name} has {n_layers} layers; probing available subset "
              f"{probe_layers} of the desired sweep {DESIRED_PROBE_LAYERS}.")
    model_probe_layers[name] = probe_layers

    # -----------------------------------------------------------------------
    # 1) Feature extraction: mean-pooled activations for every probed layer
    # -----------------------------------------------------------------------
    layer_features, model_labels_raw = extract_features(bridge, dataloader, probe_layers)

    if labels_raw is None:
        # -------------------------------------------------------------------
        # 2) Label encoding (vocabulary over all collected labels)
        # 3) Train / test split (example-level, shared across models)
        # -------------------------------------------------------------------
        # The subset was sampled without replacement, so each sequence appears
        # once: an example-level split therefore has no sequence leakage. If
        # your dataset can yield near-duplicates (same UniProt entry sampled
        # twice), split on entry IDs instead. The same split is reused for
        # every layer AND every model so the accuracies are directly
        # comparable. Computing it here (before any probe training) keeps it
        # deterministic and identical for both models.
        labels_raw = model_labels_raw
        n = len(labels_raw)
        classes = sorted(set(labels_raw.tolist()))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        labels = torch.tensor([class_to_idx[l] for l in labels_raw], dtype=torch.long)
        print(f"Classes ({len(classes)}): {classes}")

        n_train = int(TRAIN_FRACTION * n)
        perm = torch.randperm(n)
        train_idx, test_idx = perm[:n_train], perm[n_train:]
        y_train, y_test = labels[train_idx], labels[test_idx]
        print(f"Train: {n_train}  Test: {n - n_train}")
    else:
        # Same dataset, same subset, same order -> labels must match exactly;
        # otherwise the shared split would pair features with wrong labels.
        assert np.array_equal(model_labels_raw, labels_raw), \
            "Label sequence differs between models - the shared split would be invalid!"
        print("Labels identical to previous model; reusing shared encoding and split.")

    # -----------------------------------------------------------------------
    # 5) Sweep: train one probe per layer
    # -----------------------------------------------------------------------
    results = {}  # layer -> (best_test_acc, probe, history, mean, std)

    for L in probe_layers:
        print(f"\n=== {name} | Layer {L} ===")
        features = layer_features[L]
        X_train, X_test = features[train_idx], features[test_idx]

        # Standardise features (statistics from the train split only)
        X_train, X_test, feat_mean, feat_std = standardise(X_train, X_test)

        probe, history, best_acc = train_probe(X_train, y_train, X_test, y_test, len(classes))
        results[L] = (best_acc, probe, history, feat_mean, feat_std)
        print(f"{name} layer {L}: best test accuracy {best_acc:.4f}")

        # Bit annoying over time
        # # Per-layer training curves
        # curves_path = os.path.join(GRAPHICS_DIR, f"probe_{name}_layer{L}_curves(len={SEQ_LENGTH}_size={SUBSET_SIZE}).png")
        # save_training_curves(history, L, curves_path)
        # print(f"Saved {curves_path}")

    all_results[name] = results
    best_layer = max(results, key=lambda L: results[L][0])
    model_best_layer[name] = best_layer
    print(f"\n{name}: best layer {best_layer} (best test accuracy {results[best_layer][0]:.4f})")

    # -----------------------------------------------------------------------
    # 7) Per-class breakdown for this model's best layer
    # -----------------------------------------------------------------------
    per_class_breakdown(name, best_layer,
                        results[best_layer][1], layer_features[best_layer],
                        train_idx, test_idx, classes, y_test)

    # Free this model's GPU memory before booting the next one
    del layer_features, bridge, tok
    if cuda_available:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 6) Summary graphic: best test accuracy per layer, both models in one plot
# ---------------------------------------------------------------------------
summary_path = os.path.join(
    GRAPHICS_DIR,
    f"accuracy_per_layer_(len={SEQ_LENGTH}_size={SUBSET_SIZE})_{'_vs_'.join(m['name'] for m in MODELS)}.png",
)

fig, ax = plt.subplots(figsize=(9, 6))
all_layers = sorted({L for layers in model_probe_layers.values() for L in layers})

for model in MODELS:
    name = model["name"]
    r = all_results[name]
    layers = model_probe_layers[name]
    accs = [r[L][0] for L in layers]
    best = model_best_layer[name]

    ax.plot(layers, accs, marker="o", linewidth=2, color=model["color"],
            label=f"{name} (best: layer {best}, {r[best][0]:.3f})")

    # Annotate each point; first model above, second below, so the two
    # models' annotations don't overlap.
    offset = 8 if model is MODELS[0] else -14
    for L, acc in zip(layers, accs):
        ax.annotate(f"{acc:.3f}", (L, acc), textcoords="offset points",
                    xytext=(0, offset), ha="center", fontsize=8, color=model["color"])

ax.set_xlabel("Layer")
ax.set_ylabel("Best test accuracy")
ax.set_title(f"Linear probe: best test accuracy per layer (len={SEQ_LENGTH}, size={SUBSET_SIZE})")
ax.set_xticks(all_layers)
ax.legend()
fig.tight_layout()
fig.savefig(summary_path, dpi=300)
plt.close(fig)
print(f"\nSaved summary graphic {summary_path}")
for model in MODELS:
    name = model["name"]
    print(f"{name}: best layer {model_best_layer[name]} "
          f"(best test accuracy {all_results[name][model_best_layer[name]][0]:.4f})")