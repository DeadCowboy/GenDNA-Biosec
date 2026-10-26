"""
Train a linear probe on one layer of a Carbon DNA model using TransformerLens
(TransformerBridge).

Carbon (HuggingFaceBio/Carbon-{500M,3B,8B}) is a Llama-style decoder
(LlamaForCausalLM) with a custom hybrid tokenizer (6-mer DNA tokens, loaded
with trust_remote_code). DNA must be wrapped as "<dna>SEQ".

Usage
-----
# Smoke test on synthetic data (high-GC vs low-GC), no dataset needed:
python train_probe.py --demo --layer 8

# Real data: CSV with columns `sequence,label`
python train_probe.py --data my_data.csv --layer 12 --pooling mean

Requirements
------------
pip install transformer_lens transformers torch scikit-learn pandas numpy
(TransformerLens >= 3 for TransformerBridge)
"""

import argparse
import random

import numpy as np

# import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from transformer_lens.model_bridge import TransformerBridge


# ----------------------------------------------------------------------------
# Model loading via TransformerBridge
# ----------------------------------------------------------------------------
def load_carbon(repo, device, dtype, revision=None):
    # Carbon's tokenizer is custom (6-mer DNA + Qwen3 BPE), so load it ourselves
    # with trust_remote_code and hand it to the bridge.
    tok = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=True)

    try:  # preflight: is the architecture supported by this TransformerLens?
        info = TransformerBridge.check_model_support(repo)
        print("check_model_support:", info)
    except Exception as e:  # informational only
        print(f"(check_model_support unavailable: {e})")

    # Returns raw HF weights (no LayerNorm folding / centering) by default,
    # which is what you want for probing the model's actual activations.
    # Note: the bridge forces eager attention so hooks can fire, which is
    # memory-hungry for very long sequences (see --batch-size / --max-tokens).
    bridge = TransformerBridge.boot_transformers(
        repo,
        device=device,
        dtype=dtype,
        tokenizer=tok,
        revision=revision,
    )
    bridge.eval()
    return bridge, tok


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_demo_data(n=600, length=300, seed=0):
    """Synthetic smoke test: label 1 = GC-rich, label 0 = AT-rich."""
    rng = random.Random(seed)
    seqs, labels = [], []
    for i in range(n):
        gc = 0.65 if i % 2 else 0.35
        s = "".join(rng.choice("GC") if rng.random() < gc else rng.choice("AT") for _ in range(length))
        seqs.append(s)
        labels.append(i % 2)
    return seqs, labels


def load_csv(path):
    # df = pd.read_csv(path)
    assert {"sequence", "label"} <= set(df.columns), "CSV needs `sequence` and `label` columns"
    return df["sequence"].astype(str).str.upper().tolist(), df["label"].tolist()


# ----------------------------------------------------------------------------
# Activation extraction
# ----------------------------------------------------------------------------
@torch.no_grad()
def extract_features(bridge, tok, seqs, layer, hook, pooling, batch_size, max_tokens, device):
    # Canonical bridge hook names: blocks.{i}.hook_out is the residual stream
    # leaving block i (legacy alias: hook_resid_post). Others: hook_in
    # (hook_resid_pre), attn.hook_out, mlp.hook_out.
    hook_name = f"blocks.{layer}.{hook}"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    feats = []
    use_stop = True

    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        ids = [tok(f"<dna>{s}", add_special_tokens=False)["input_ids"][:max_tokens] for s in batch]
        L = max(len(x) for x in ids)
        # Right padding is safe: the model is causal, so real positions never
        # attend to the pad tokens that come after them.
        tokens = torch.full((len(ids), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(ids), L), dtype=torch.bool)
        for j, x in enumerate(ids):
            tokens[j, : len(x)] = torch.tensor(x)
            mask[j, : len(x)] = True
        tokens, mask = tokens.to(device), mask.to(device)

        # Passing a token tensor skips the bridge's own tokenization, so no BOS
        # is added -- the input is exactly "<dna>SEQ" as Carbon expects.
        # stop_at_layer skips all computation after the probed layer.
        try:
            if use_stop:
                _, cache = bridge.run_with_cache(tokens, names_filter=hook_name, stop_at_layer=layer + 1)
            else:
                _, cache = bridge.run_with_cache(tokens, names_filter=hook_name)
        except (NotImplementedError, TypeError) as e:
            print(f"\n  stop_at_layer not usable ({e}); running the full model")
            use_stop = False
            _, cache = bridge.run_with_cache(tokens, names_filter=hook_name)

        acts = cache[hook_name].float()  # [batch, pos, d_model]
        assert torch.isfinite(acts).all(), "non-finite activations (try dtype float32)"

        if pooling == "mean":
            m = mask.unsqueeze(-1).float()
            pooled = (acts * m).sum(1) / m.sum(1)
        elif pooling == "last":
            last = mask.sum(1) - 1
            pooled = acts[torch.arange(acts.shape[0]), last]
        else:
            raise ValueError(pooling)
        feats.append(pooled.cpu())
        print(f"  extracted {min(i + batch_size, len(seqs))}/{len(seqs)}", end="\r")
    print()
    return torch.cat(feats)


# ----------------------------------------------------------------------------
# Probe
# ----------------------------------------------------------------------------
def train_probe(Xtr, ytr, Xva, yva, n_classes, epochs, lr, wd, device):
    probe = nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    Xtr, ytr, Xva, yva = Xtr.to(device), ytr.to(device), Xva.to(device), yva.to(device)

    best_acc, best_state = -1, None
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(len(Xtr), device=device)
        for k in range(0, len(Xtr), 256):
            idx = perm[k : k + 256]
            opt.zero_grad()
            loss_fn(probe(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            acc = (probe(Xva).argmax(-1) == yva).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  val acc {acc:.3f}")
    probe.load_state_dict(best_state)
    return probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceBio/Carbon-500M")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--layer", type=int, required=True, help="0-indexed transformer block")
    ap.add_argument("--hook", default="hook_out", help="hook inside the block: hook_out (resid post), hook_in (resid pre), " "attn.hook_out, mlp.hook_out")
    ap.add_argument("--pooling", choices=["mean", "last"], default="mean")
    ap.add_argument("--data", help="CSV with `sequence,label` columns")
    ap.add_argument("--demo", action="store_true", help="use synthetic GC-content data")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8192, help="Carbon context is 8192 tokens")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out", default="probe.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if args.demo:
        seqs, labels = make_demo_data(seed=args.seed)
    elif args.data:
        seqs, labels = load_csv(args.data)
    else:
        ap.error("pass --data FILE.csv or --demo")

    print(f"Loading {args.model} via TransformerBridge on {device}...")
    bridge, tok = load_carbon(args.model, device, dtype, args.revision)
    n_layers = getattr(bridge.cfg, "n_layers", None)
    if n_layers is not None:
        print(f"Model has {n_layers} layers")
        assert 0 <= args.layer < n_layers, f"layer must be in [0, {n_layers - 1}]"

    print(f"Extracting blocks.{args.layer}.{args.hook} ({args.pooling} pooled)...")
    X = extract_features(bridge, tok, seqs, args.layer, args.hook, args.pooling, args.batch_size, args.max_tokens, device)
    classes = sorted(set(labels))
    y = torch.tensor([classes.index(l) for l in labels])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.15, random_state=args.seed, stratify=ytr)

    # Standardise using train statistics only (avoids leakage).
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

    print("Training probe...")
    probe = train_probe(Xtr, ytr, Xva, yva, len(classes), args.epochs, args.lr, args.weight_decay, device)

    probe.eval()
    with torch.no_grad():
        pred = probe(Xte.to(device)).argmax(-1).cpu().numpy()
    print(f"\nTest accuracy: {accuracy_score(yte, pred):.4f}")
    print(f"Test macro-F1: {f1_score(yte, pred, average='macro'):.4f}")
    print(f"Majority baseline: {np.bincount(yte.numpy()).max() / len(yte):.4f}")

    torch.save({"state_dict": probe.state_dict(), "mu": mu, "sd": sd, "classes": classes, "layer": args.layer, "hook": args.hook, "pooling": args.pooling, "model": args.model}, args.out)
    print(f"Saved probe to {args.out}")


if __name__ == "__main__":
    main()
