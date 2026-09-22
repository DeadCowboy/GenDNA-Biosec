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
import matplotlib.pyplot as plt

import inspect

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


MAX_LENGTH = 1500
BIN_SIZE = 24
TEMP = 1.0
TOP_P = 0.9
MASKING_RATIO = 0.5
BATCH_SIZE = 200

SUBSET_SIZE = BATCH_SIZE * 30  # Number of training examples in the subset


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


# Function to run evaluation for a specific EC range
def run_ec_evaluation(ec_range):
    print(f"\n--- Evaluating EC Range {ec_range} ---")

    # Loading Dataset
    dataset = SwissProtDataset(masking_ratio=MASKING_RATIO, seq_length=MAX_LENGTH, length_multiple=6, ec_range=ec_range)  # Disable Masking with ratio 0

    indices = torch.randperm(len(dataset))[:SUBSET_SIZE].tolist()
    true_dataset_size = len(dataset)
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn)

    # Eval
    distances_list = []

    def for_each_run(out, label_dict, input_ids):
        gen_seq = tok.decode(out, skip_special_tokens=True)  # Have some overlap so you can see the sequences are infact correctly aligned
        compare_input_seq = label_dict["input_sequence"]

        end_of_input = input_ids.numel() * 6 - 6

        assert gen_seq[:end_of_input] == compare_input_seq[:end_of_input], ValueError(
            f"Generated sequence does not match input sequence up to the length of the input ids. \n Generated: {gen_seq[:end_of_input]} \n Input: {compare_input_seq[:end_of_input]}"
        )

        generated_seq = gen_seq[end_of_input:]
        gt_seq = compare_input_seq[end_of_input:]  # Ground truth sequence
        assert len(generated_seq) == len(gt_seq), ValueError(f"Generated sequence length {len(generated_seq)} does not match ground truth sequence length {len(gt_seq)}")

        start = 0
        end = BIN_SIZE
        distances = []
        while end < len(generated_seq):
            gen_bin = generated_seq[start:end]
            gt_bin = gt_seq[start:end]
            distance = Levenshtein.distance(gen_bin, gt_bin)
            distances.append(distance)
            # print(f"Generated: {gen_bin} | Ground Truth: {gt_bin} | Distance: {distance}")
            start += BIN_SIZE
            end += BIN_SIZE
        distances_list.append(distances)

    for input_ids, attention_mask, label_dict in tqdm(dataloader):
        new_tokens: int = len(label_dict[0]["input_sequence"]) // 6 - input_ids[0].numel() + 1
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=new_tokens,
            do_sample=False,
        )
        for i in range(BATCH_SIZE):
            for_each_run(out[i], label_dict[i], input_ids[i])

    # Plot
    distances_array = np.array(distances_list)
    lower = np.percentile(distances_array, 25, axis=0)
    upper = np.percentile(distances_array, 75, axis=0)
    mean = distances_array.mean(axis=0)

    x = np.arange(distances_array.shape[1])
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean, label="Mean", linewidth=2)
    plt.fill_between(x, lower, upper, alpha=0.25, label="25–75%")
    plt.xlabel(f"Steps of {BIN_SIZE} basepairs")
    plt.ylabel("Levenshtein Distance")
    plt.title(f"Generation Quality Over Time ({true_dataset_size} samples, Masking Ratio: {MASKING_RATIO}, EC Range: {ec_range})")
    plt.legend()

    # Save the figure
    output_dir = "/home/lkammeyer/Desktop/Master_Thesis_link/GenDNA-Biosec/carbon/EC_evaluation_figures"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"ec_range_{ec_range}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to {filepath}")
    return filepath


# Run evaluation for EC ranges 1-7
ec_ranges = ["1", "2", "3", "4", "5", "6", "7"]
saved_files = []

for ec_range in ec_ranges:
    saved_file = run_ec_evaluation(ec_range)
    saved_files.append(saved_file)

print("\nAll evaluations completed!")
print("Saved files:")
for f in saved_files:
    print(f"  - {f}")
