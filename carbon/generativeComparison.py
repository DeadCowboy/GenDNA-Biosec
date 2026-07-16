from dataset.Protein.EnzymeDatasetAutoreg import SwissProtDataset

import sys
import os

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import contextlib
import Levenshtein
from difflib import ndiff
from tqdm import tqdm

import inspect

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


SEQ_LENGTH = 600
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
dataset = SwissProtDataset(masking_ratio=MASKING_RATIO, seq_length=SEQ_LENGTH, length_multiple=6)  # Disable Masking with ratio 0

indices = torch.randperm(len(dataset))[:SUBSET_SIZE].tolist()
subset = Subset(dataset, indices)
dataloader = DataLoader(subset, shuffle=False, collate_fn=dataset.collate_fn)


for input_ids, attention_mask, label_dict in dataloader:
    # Confirm dataset masks correctly
    # print(tok.decode(input_ids[0], skip_special_tokens=True))

    # print("Input")
    # print(label_dict[0]["input_sequence"][: len(label_dict[0]["input_sequence"]) // 2])

    new_tokens = len(label_dict[0]["input_sequence"]) // 6 - input_ids.numel()

    # Are these real params? temperature=TEMP, top_p=TOP_P, do_sample=True, return_dict_in_generate=True, output_scores=True
    out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=new_tokens)
    out_seq = tok.decode(out[0], skip_special_tokens=True)  # Have some overlap so you can see the sequences are infact correctly alignt
    compare_input_seq = label_dict[0]["input_sequence"]

    # [input_ids.numel() - 1 :]
    # [(input_ids.numel() - 2) * 6 :]

    # print("Input: ", compare_input_seq)
    # print("\n \n \n")
    # print("Generated: ", out_seq)

    # for line in ndiff([out_seq], [compare_input_seq]):
    #     print(line)
    # print("\n")
    print(f"Distance: {Levenshtein.distance(compare_input_seq[:input_ids.numel()], out_seq[:input_ids.numel()])}")
