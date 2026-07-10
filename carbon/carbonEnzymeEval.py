from dataset.Protein.EnzymeDatasetAutoreg import SwissProtDataset
import sys
import os
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import contextlib
import Levenshtein
from tqdm import tqdm

import inspect

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


MAX_LENGTH = 600
STEPS = 25
TEMP = 1.0
TOP_P = 0.9
MASKING_RATIO = 0.5

SUBSET_SIZE = 1  # Number of training examples in the subset


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
repo = "HuggingFaceBio/Carbon-3B"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16)
if torch.cuda.is_available():
    model = model.cuda().eval()
else:
    model = model.eval()


# Loading Dataset
dataset = SwissProtDataset(masking_ratio=MASKING_RATIO, max_length=MAX_LENGTH, length_multiple=6)  # Disable Masking with ratio 0

indices = torch.randperm(len(dataset))[:SUBSET_SIZE].tolist()
subset = Subset(dataset, indices)
dataloader = DataLoader(subset, shuffle=False, collate_fn=dataset.collate_fn)


# Eval


for input_ids, attention_mask, label_dict in dataloader:
    # print(input_ids.numel())
    # Wrap a DNA prompt with the <dna> tag (the model is trained with this format).
    # DNA length should be a multiple of 6 — see the Tokenizer section below.
    # dna_prompt = "ATGCGCTAGCTACGATCGATCGTAGCTAGCTAGCTAGCTACG"  # 42 bp = 7 × 6-mer
    # prompt = f"<dna>{dna_prompt}"
    # print("Doing Stuff")
    # inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=16,
        do_sample=False,
    )
    print("Label Dict")
    print(label_dict)

    print("\n \nDecode")
    print(tok.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True))

    print("Input Sequence After target length")
    print(label_dict[0]["input_sequence"][label_dict[0]["target_length"] : label_dict[0]["target_length"] + 16 * 6])
