from dataset.Protein.EnzymeDataset import SwissProtDataset
import sys
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


MAX_LENGTH = 600
STEPS = 25
TEMP = 1.0
TOP_P = 0.9
MASKING_RATIO = 0.0

SUBSET_SIZE = 300  # Number of training examples in the subset

# Instanciate Dataset with Custom Masking transform
dataset = SwissProtDataset(masking_ratio=MASKING_RATIO, max_length=MAX_LENGTH)

# TODO: Remove for full run!
indices = torch.randperm(len(dataset))[:SUBSET_SIZE].tolist()
subset = Subset(dataset, indices)


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

config = MDMGenerationConfig(
    mask_token_id=tokenizer.mask_token_id,
    max_length=MAX_LENGTH,
    steps=STEPS,
    temperature=TEMP,
    top_p=TOP_P,
    alg="random",
    num_return_sequences=1,
    return_dict_in_generate=True,
)


def genConfigStr():
    return f"Max Length: {MAX_LENGTH} \n STEPS: {STEPS} \n TEMP: {TEMP} \n TOP_P: {TOP_P}\n MASKING_RATIO: {MASKING_RATIO}\n SUBSET_SIZE: {SUBSET_SIZE}"


def count_runs(text):
    if not text:
        return []

    run_counts = {}

    current_run = 1

    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            current_run += 1
        else:
            run_counts[current_run] = run_counts.get(current_run, 0) + 1
            current_run = 1

    # Count the final run
    run_counts[current_run] = run_counts.get(current_run, 0) + 1

    max_run = max(run_counts)
    result = [0] * max_run

    for run_length, count in run_counts.items():
        result[run_length - 1] = count

    return result


# Instanciate Dataloader
dataloader = DataLoader(subset, shuffle=False, collate_fn=dataset.collate_fn)

length_list = []
for input_ids, label_batch in dataloader:
    with torch.no_grad():
        input_ids = input_ids.to(device)
        outputs = model.diffusion_generate(inputs=input_ids, generation_config=config)

        for i, seq in enumerate(outputs.sequences):
            decoded = tokenizer.decode(seq, skip_special_tokens=True).replace(" ", "")
            label = label_batch[0]
            orig_seq = label["original_sequence"]
            masked_indices = label["masked_indices"]
            print(f" Runs Decoded: {count_runs(decoded)} \n \n Runs Original Sequence: {count_runs(orig_seq[:len(decoded)])}")
            length_list.append(len(decoded))

    # Kill switch
    exit()

print(f"Max Length: {MAX_LENGTH} \n STEPS: {STEPS} \n TEMP: {TEMP} \n TOP_P: {TOP_P}\n MASKING_RATIO: {MASKING_RATIO}\n SUBSET_SIZE: {SUBSET_SIZE} -->", length_list)
# print(f">{i} {len(decoded)}  \n {MAX_LENGTH + len(masked_indices) * 5} \n {len(decoded) - (MAX_LENGTH + len(masked_indices) * 5)} \n ")


# Complete some enzyme sequences and meassure how good it is
