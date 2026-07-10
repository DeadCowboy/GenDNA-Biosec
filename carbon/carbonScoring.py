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


def score(seq: str) -> float:
    """Mean log-prob per DNA token of `seq` (single sequence, no padding)."""
    text = f"<dna>{seq}</dna>"
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    logits = model(ids).logits[:, :-1, :]
    targets = ids[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logp.mean().item()


def print_score(seq: str):
    """Prints the log-probabilities of each token in `seq`."""
    text = f"<dna>{seq}</dna>"
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    logits = model(ids).logits[:, :-1, :]
    targets = ids[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    for token, lp in zip(ids, logp[0].tolist()):
        print(f"Token: {tok.decode(token, skip_special_tokens=True)}, Log-Prob: {lp:.4f}")


def score_last_n(seq: str, n: int) -> float:
    """Mean log-prob per DNA token of the last `n` tokens of `seq` (single sequence, no padding)."""
    text = f"<dna>{seq}</dna>"
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    logits = model(ids).logits[:, :-1, :]
    targets = ids[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logp[:, -n:].mean().item()


def print_last_n(seq: str, n: int):
    """Prints the last `n` tokens of `seq` and their log-probabilities."""
    text = f"<dna>{seq}</dna>"
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    logits = model(ids).logits[:, :-1, :]
    targets = ids[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    last_n_tokens = ids[0, -n:].tolist()
    last_n_logp = logp[0, -n:].tolist()
    decoded_tokens = [tok.decode(token, skip_special_tokens=True) for token in last_n_tokens]
    for token, lp in zip(decoded_tokens, last_n_logp):
        print(f"Token: {token}, Log-Prob: {lp:.4f}")


def scramble_last_n(seq: str, n: int):
    """
    Make the last n Letters of a sequence random, and return the new sequence.
    """
    if n > len(seq):
        raise ValueError(f"n ({n}) is greater than the length of the sequence ({len(seq)})")
    prefix = seq[:-n]
    suffix = seq[-n:]
    suffix_list = list(suffix)
    np.random.shuffle(suffix_list)
    scrambled_suffix = "".join(suffix_list)
    return prefix + scrambled_suffix


def scramble(seq: str):
    """
    Make the sequence random, and return the new sequence.
    """
    seq_list = list(seq)
    np.random.shuffle(seq_list)
    scrambled_seq = "".join(seq_list)
    return scrambled_seq


seq = "CAGGCACAGACCATGCAGACTCCAGTTCCTCCTGCTGCTCCTGATGCTGGGATGCGTCCTGATGATGGTGGCGATGTTGCACCCTCCCCACCACACCCTGCACCAGACTGTCACAGCCCAAGCCAGCAAGCACAGCCCTGAAGCCAGGTACCGCCTGGACTTTGGGGAATCCCAGGATTGGGTACTGGAAGCTGAGGATGAGGGTGAAGAGTACAGCCCTCTGGAGGGCCTGCCACCCTTTATCTCACTGCGGGAGGATCAGCTGCTGGTGGCCGTGGCCTTACCCCAGGCCAGAAGGAACCAGAGCCAGGGCAGGAGAGGTGGGAGCTACCGCCTCATCAAGCAGCCAAGGAGGCAGGATAAGGAAGCCCCAAAGAGGGACTGGGGGGCTGATGAGGACGGGGAGGTGTCTGAAGAAGAGGAGTTGACCCCGTTCAGCCTGGACCCACGTGGCCTCCAGGAGGCACTCAGTGCCCGCATCCCCCCTCCAGAGGGCTCTGCCCGAGGTGCGGCACCCACTGTGTCTGCAGCAGCACCCTCAGGACAGCCTGCCACAGCCAGCGTCATCCTCTGTTTCCATGATGAGGCCTGGTCCATCTCCTGCGGACTGTACACAGCATCCTCGACACAGTGCCCAGGGCTTCCTGAAGGAGATCATCCTCGTGGACGACCTCAGCCAGCAAGGACAACTCAAGTCTGCTCTCAGCGAATATGTGGCCAGGCTGGAGGGGGTGAAGTTACTCAGGAGCAACAAGAGGCTGGGTGCCATCAGGGCCCGGATGCTGGAGCCACCAGAGCCACCGGGGATGTGCTCGTCTTCATGGATGCCCACTGCGAGTGCCACCCAGGCTGGCTGGAGCCCCTCCTCAGCAGAATAGCTGGTGACAGGAGCCGAGTGGTATCTCCGGTGATAGATGTGATTGACTGGAAGACTTTCCAGTATTACCCCTCAAGGGACCTGCAGCGTGGGGTGTTGGACTGGAAGCTGGATTTCCACTGGGAACCTTTGCCAGAGCATGTGAGGAAGGCCCTCCAGTCCCCCATAAGCCCCATCAGGAGCCCTGTGGTGCCCGGAGAGGTGGTGGCCATGGACAGACATTACTTCCAAAACACTGGAGCGTATGACTCTCTTATGTCGCTGCGAGGTGGTGAAAACCTCGAACTGTCTTTCAAGGCCTGGCTCTGTGGTGGCTCTGTTGAAATCCTTCCCTGCTCTCGGGTAGGACACATCTACCAAAATCAGGATTCCCATTCCCCCCTCGACCAGGAGACCACCCTGAGGAACAGGGTTCGCATTGCTGAGACCTGGCTGGGGTCATTCAAAGAAACCTTCTACAAGCATAGCCCAGAGGCCTTCTCCTTGAGCAAGGCTGAGAAGCCAGACTGCATGGAACGCTTGCAGCTGCAAAGGAGACTGGGTTGTCGGACATTCCACTGGTTTCTGGCTAATGTCTACCCTGAGCTGTACCCATCTGAACCCAGGCCCAGTTTCTCTGGAAAGCTCCACAACACTGGACTTGGGCTCTGTGCAGACTGCCAGGCAGAAGGGGACATCCTGGGCTGTCCCATGGTGTTGGCTCCTTGCAGTGACAGCCGGCAGCAACAGTACCTGCAGCACACCAGCAGGAAGGAGATTCACTTTGGCAGCCCACAGCACCTGTGCTTTGCTGTCAGGCAGGAGCAGGTGATTCTTCAGAACTGCACGGAGGAAGGCCTGGCCATCCACCAGCAGCACTGGGACTTCCAGGAGAATGGGATGATTGTCCACATTCTTTCTGGGAAATGCATGGAAGCTGTGGTGCAAGAAAACAATAAAGATTTGTACCTGCGTCCGTGTGATGGAAAAGCCCGCCAGCAGTGGCGTTTTGACCAGATCAATGCTGTGGATGAACGATG"
print("Vanilla")
print(score(seq))
print("Shuffeled")
print(score(scramble(seq)))


buffer = "CCCAAGCCAGCAAGCACAGCCCTGAAGCCAGGTACCGCCTGGACTTTGGGGAATCCCAGGATTGGGTACTGGAAGCTGAGGATGAGGGTGAAGAGTACAGCCCTCTGGAGGGCCTGCCACCCTTTATCTCACTGCGGGAGGATCAGCTGCTGGTGGCCGTGGCCTTACCCCAGGCCAGAAGGAACCAGAGCCAGGGCAGGAGAGGTGGGAGCTACCGCCTCATCAAGCAGCCAAGGAGGCAGGATAAGGAAGCCCCAAAGAGGGACTGGGGGGCTGATGAGGACGGGGAGGTGTCTGAAGAAGAGGAGTTGACCCCGTTCAGCCTGGACCCACGTGGCCTCCAGGAGGCACTCAGTGCCCGCATCCCCCCTCCAGAGGGCTCTGCCCGAGGTGCGGCACCCACTGTGTCTGCAGCAGCACCCTCAGGACAGCCTGCCACAGCCAGCGTCATCCTCTGTTTCCATGATGAGGCCTGGTCCATCTCCTGCGGACTGTACACAGCATCCTCGACACAGTGCCCAGGGCTTCCTGAAGGAGATCATCCTCGTGGACGACCTCAGCCAGCAAGGACAACTCAAGTCTGCTCTCAGCGAATATGTGGCCAGGCTGGAGGGGGTGAAGTTACTCAGGAGCAACAAGAGGCTGGGTGCCATCAGGGCCCGGATGCTGGAGCCACCAGAGCCACCGGGGATGTGCTCGTCTTCATGGATGCCCACTGCGAGTGCCACCCAGGCTGGCTGGAGCCCCTCCTCAGCAGAATAGCTGGTGACAGGAGCCGAGTGGTATCTCCGGTGATAGATGTGATTGACTGGAAGACTTTCCAGTATTACCCCTCAAGGGACCTGCAGCGTGGGGTGTTGGACTGGAAGCTGGATTTCCACTGGGAACCTTTGCCAGAGCATGTGAGGAAGGCCCTCCAGTCCCCCATAAGCCCCATCAGGAGCCCTGTGGTGCCCGGAGAGGTGGTGGCCATGGACAGACATTACTTCCAAAACACTGGAGCGTATGACTCTCTTATGTCGCTGCGAGGTGGTGAAAACCTCGAACTGTCTTTCAAGGCCTGGCTCTGTGGTGGCTCTGTTGAAATCCTTCCCTGCTCTCGGGTAGGACACATCTACCAAAATCAGGATTCCCATTCCCCCCTCGACCAGGAGACCACCCTGAGGAACAGGGTTCGCATTGCTGAGACCTGGCTGGGGTCATTCAAAGAAACCTTCTACAAGCATAGCCCAGAGGCCTTCTCCTTGAGCAAGGCTGAGAAGCCAGACTGCATGGAACGCTTGCAGCTGCAAAGGAGACTGGGTTGTCGGACATTCCACTGGTTTCTGGCTAATGTCTACCCTGAGCTGTACCCATCTGAACCCAGGCCCAGTTTCTCTGGAAAGCTCCACAACACTGGACTTGGGCTCTGTGCAGACTGCCAGGCAGAAGGGGACATCCTGGGCTGTCCCATGGTGTTGGCTCCTTGCAGTGACAGCCGGCAGCAACAGTACCTGCAGCACACCAGCAGGAAGGAGATTCACTTTGGCAGCCCACAGCACCTGTGCTTTGCTGTCAGGCAGGAGCAGGTGATTCTTCAGAACTGCACGGAGGAAGGCCTGGCCATCCACCAGCAGCACTGGGACTTCCAGGAGAATGGGATGATTGTCCACATTCTTTCTGGGAAATGCATGGAAGCTGTGGTGCAAGAAAACAATAAAGATTTGTACCTGCGTCCGTGTGATGGAAAAGCCCGCCAGCAGTGGCGTTTTGACCAGATCAATGCTGTGGATGAACGATG"
