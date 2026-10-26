# Initial run for the linear probe to proof the mehtod

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
import matplotlib.pyplot as plt

import inspect

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


SEQ_LENGTH = 1000
MASKING_RATIO = 0.5
BATCH_SIZE = 60

SUBSET_SIZE = BATCH_SIZE * 100  # Number of training examples in the subset
print(f"Sample size: {SUBSET_SIZE}")


# Params for Probe
layer = 15  # Layer to probe
hook = "hook_out"  # Hook to probe


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
# memory-hungry for very long sequences (see --batch-size / --max-tokens).
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
bridge = TransformerBridge.boot_transformers(
    repo,
    device=device,
    dtype=dtype,
    tokenizer=tok,
)
bridge.eval()


# Loading Dataset
dataset = SwissProtDataset(masking_ratio=MASKING_RATIO, seq_length=SEQ_LENGTH, length_multiple=6, ec_range="2")

indices = torch.randperm(len(dataset))[:SUBSET_SIZE].tolist()
subset = Subset(dataset, indices)
dataloader = DataLoader(subset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn)


for batch_idx, (input_ids, attention_mask, label_dict) in enumerate(tqdm(dataloader, desc="Processing batches")):
    new_tokens: int = len(label_dict[0]["input_sequence"]) // 6 - input_ids[0].numel() + 1
    hook_name = f"blocks.{layer}.{hook}"
    _, cache = bridge.run_with_cache(input_ids, names_filter=hook_name, stop_at_layer=layer + 1)
    activations = cache[hook_name].float()  # [batch, pos, d_model]
    assert torch.isfinite(activations).all()  # If this fails something went badly wrong

    # Mean pooling along sequence dimension
    pooled = activations.mean(dim=1)  # [batch, d_model]
    print(pooled.shape)

    # Labels
    labels = [label_dict[i]["ec_number"][0] for i in range(len(label_dict))]
    break
