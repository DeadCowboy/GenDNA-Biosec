from dataset.Protein.EnzymeDatasetDiffusion import SwissProtDataset
import sys
import os
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForMaskedLM
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

SUBSET_SIZE = 100  # Number of training examples in the subset


# Cuda
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

if cuda_available:
    print("GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Setting up Model
print("Initializing model and tokenizer...")
model_name = "Hengchang-Liu/D3LM-from-nt"
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
MDMGenerationConfig = getattr(sys.modules[type(model).__module__], "MDMGenerationConfig")


# print(inspect.getcallargs(MDMGenerationConfig.__init__, self=MDMGenerationConfig))
# exit()

# print(config.to_dict())
# exit()

res = []
for MASKING_RATIO in tqdm(range(0, 10)):
    MASKING_RATIO = MASKING_RATIO / 10
    # Instanciate Dataset with Custom Masking transform
    dataset = SwissProtDataset(masking_ratio=MASKING_RATIO, max_length=MAX_LENGTH)

    # TODO: Remove for full run!
    # indices = torch.randperm(len(dataset))[:SUBSET_SIZE].tolist()
    # subset = Subset(dataset, indices)

    config = MDMGenerationConfig(
        mask_token_id=tokenizer.mask_token_id,
        max_length=MAX_LENGTH,
        steps=STEPS,
        temperature=TEMP,
        top_p=TOP_P,
        alg="random",
        num_return_sequences=1,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=0,
    )

    def genConfigStr():
        return f"Max Length: {MAX_LENGTH} \n STEPS: {STEPS} \n TEMP: {TEMP} \n TOP_P: {TOP_P}\n MASKING_RATIO: {MASKING_RATIO}\n SUBSET_SIZE: {SUBSET_SIZE}"

    # Instanciate Dataloader
    dataloader = DataLoader(dataset, shuffle=False, collate_fn=dataset.collate_fn)

    # print(f"\n \n MASKING_RATIO: { MASKING_RATIO}")
    levenshtein_dist = []
    for sample, label_dict in dataloader:
        with torch.no_grad():
            input_ids = sample[0].to(device)
            sys.stdout = open(os.devnull, "w")  # Make the model shut up
            output = model.diffusion_generate(inputs=input_ids, generation_config=config)
            sys.stdout = sys.__stdout__
            print("Confirm!")
            out = output.sequences[0]
            decoded = tokenizer.decode(out, skip_special_tokens=True).replace(" ", "")
            # print(f"len decoded: {len(decoded)}    len text input: {len(label_dict[0]['trunc_input'])}")
            levenshtein_dist.append(Levenshtein.distance(decoded, label_dict[0]["trunc_input"]))

    # print(f"len orig: {len(label_dict[0]['trunc_input'])}    len generated: {len(decoded)}    len / leven: {MAX_LENGTH / np.average(levenshtein_dist)}")
    # res.append([f"\n \n MASKING_RATIO: { MASKING_RATIO} \n   len orig: {len(label_dict[0]['trunc_input'])}    len generated: {len(decoded)}    len / leven: {MAX_LENGTH / np.average(levenshtein_dist)} \n {levenshtein_dist} \n \n" ])
    res.append(levenshtein_dist)

print(res)
