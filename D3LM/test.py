import sys
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM



import warnings
warnings.filterwarnings("ignore")


print("Initializing model and tokenizer...")
model_name = "Hengchang-Liu/D3LM-from-nt"
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# print("Import Config")
# # Import MDMGenerationConfig from the model's auto-downloaded module
# MDMGenerationConfig = getattr(sys.modules[type(model).__module__], "MDMGenerationConfig")

# print("Generating sequences...")
# # Unconditional generation: create a fully-masked prompt of desired length
# length = 200
# input_ids = torch.full((1, length), tokenizer.mask_token_id, dtype=torch.long)

# config = MDMGenerationConfig(
#     mask_token_id=tokenizer.mask_token_id,
#     max_length=length,
#     steps=50,
#     temperature=1.0,
#     top_p=0.9,
#     alg="random",
#     num_return_sequences=4,
#     return_dict_in_generate=True,
# )

# with torch.no_grad():
#     outputs = model.diffusion_generate(inputs=input_ids, generation_config=config)

# print("Running")
# for i, seq in enumerate(outputs.sequences):
#     print(f">{i} {tokenizer.decode(seq, skip_special_tokens=True).replace(' ', '')}")

print(f"Target: {tokenizer.pad_token_id}")