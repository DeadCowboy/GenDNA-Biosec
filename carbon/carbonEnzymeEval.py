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
# MASKING_RATIO = 0.5

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
# model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval()


# Loading Dataset
dataset = SwissProtDataset(masking_ratio=0, max_length=MAX_LENGTH, length_multiple=6, do_masking=False)  # Disable Masking with ratio 0

indices = torch.randperm(len(dataset))[:SUBSET_SIZE].tolist()
subset = Subset(dataset, indices)
dataloader = DataLoader(subset, shuffle=False, collate_fn=dataset.collate_fn)


# Eval


for sample, label_dict in dataloader:
    # Wrap a DNA prompt with the <dna> tag (the model is trained with this format).
    # DNA length should be a multiple of 6 — see the Tokenizer section below.
    # dna_prompt = "ATGCGCTAGCTACGATCGATCGTAGCTAGCTAGCTAGCTACG"  # 42 bp = 7 × 6-mer
    # prompt = f"<dna>{dna_prompt}"
    # print("Doing Stuff")
    # inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    # out = model.generate(
    #     **inputs,
    #     max_new_tokens=64,
    #     do_sample=False,
    # )
    # print(tok.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True))

    print(sample)
