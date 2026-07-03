from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "HuggingFaceBio/Carbon-3B"

tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval()

# # Wrap a DNA prompt with the <dna> tag (the model is trained with this format).
# # DNA length should be a multiple of 6 — see the Tokenizer section below.
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

print(tok.__dict__)
