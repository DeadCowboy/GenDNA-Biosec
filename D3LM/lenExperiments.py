from dataset.EnzymeDataset import SwissProtDataset
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
MASKING_RATIO = 0.5

SUBSET_SIZE = 1  # Number of training examples in the subset

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


# Instanciate Dataloader
dataloader = DataLoader(subset, shuffle=False, collate_fn=dataset.collate_fn)


gen_lengths = []
for sample in dataloader:
    with torch.no_grad():
        input_ids = sample[0].to(device)
        output = model.diffusion_generate(inputs=input_ids, generation_config=config)
        out = output.sequences[0]
        decoded = tokenizer.decode(out, skip_special_tokens=True).replace(" ", "")
        gen_lengths.append(len(decoded))
print(f"Target Length: {MAX_LENGTH}     Generated ID Length:{len(out)}       Min Generated Text Length:{min(gen_lengths)}        Max Generated Text Length:{max(gen_lengths)}")