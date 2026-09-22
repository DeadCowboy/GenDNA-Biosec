# The generated sequence is compared to the ground truth sequence in bins of 24 basepairs. The Levenshtein distance is calculated for each bin and stored in a list.
# The mean and interquartile range of the distances are then plotted over time to visualize the generation quality of the model.


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
import matplotlib.pyplot as plt

import inspect

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


SEQ_LENGTH = 2048
MASKING_RATIO = 0.75
BATCH_SIZE = 128

SUBSET_SIZE = BATCH_SIZE * 300  # Number of training examples in the subset
print(f"Sample size: {SUBSET_SIZE}")


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
dataloader = DataLoader(subset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn)


distances_list = []


def for_each_run(out, label_dict, input_ids):

    # Are these real params? temperature=TEMP, top_p=TOP_P, do_sample=True, return_dict_in_generate=True, output_scores=True
    # out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=new_tokens)
    out_seq = tok.decode(out, skip_special_tokens=True)  # Have some overlap so you can see the sequences are infact correctly alignt
    compare_input_seq = label_dict["input_sequence"]

    end_of_input = input_ids.numel() * 6 - 6

    assert out_seq[:end_of_input] == compare_input_seq[:end_of_input], ValueError(
        f"Generated sequence does not match input sequence up to the length of the input ids. \n Generated: {out_seq[:end_of_input]} \n Input: {compare_input_seq[:end_of_input]}"
    )

    generated_seq = out_seq[end_of_input:]
    gt_seq = compare_input_seq[end_of_input:]
    assert len(generated_seq) == len(gt_seq), ValueError(f"Generated sequence length {len(generated_seq)} does not match ground truth sequence length {len(gt_seq)}")

    bin_size = 24
    start = 0
    end = bin_size
    distances = []
    while end < len(generated_seq):
        gen_bin = generated_seq[start:end]
        gt_bin = gt_seq[start:end]
        distance = Levenshtein.distance(gen_bin, gt_bin)
        distances.append(distance)
        # print(f"Generated: {gen_bin} | Ground Truth: {gt_bin} | Distance: {distance}")
        start += bin_size
        end += bin_size

    distances_list.append(distances)
    # print(f"Distances: {distances}")
    # print(f"Distance: {Levenshtein.distance(compare_input_seq, out_seq)}   Len generated: {len(out_seq)}    Len input: {len(compare_input_seq)}")


for input_ids, attention_mask, label_dict in tqdm(dataloader):
    new_tokens = len(label_dict[0]["input_sequence"]) // 6 - input_ids[0].numel() + 1
    out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=new_tokens, do_sample=False)
    for i in range(BATCH_SIZE):
        for_each_run(out[i], label_dict[i], input_ids[i])

print(distances_list)
distances_array = np.array(distances_list)


# Plot

lower = np.percentile(distances_array, 25, axis=0)
upper = np.percentile(distances_array, 75, axis=0)
mean = distances_array.mean(axis=0)

x = np.arange(distances_array.shape[1])
plt.plot(x, mean, label="Mean")
plt.fill_between(x, lower, upper, alpha=0.25, label="25–75%")
plt.xlabel(f"Steps of {24} basepairs")
plt.ylabel("Levenshtein Distance")
plt.title(f"Generation Quality Over Time ({SUBSET_SIZE} samples, Masking Ratio: {MASKING_RATIO})")
plt.legend()
plt.show()
# plt.savefig(f"/home/lkammeyer/Documents/Uni/Master_Thesis/output/carbon/GenComp_{SUBSET_SIZE}_{MASKING_RATIO}_24.png")
