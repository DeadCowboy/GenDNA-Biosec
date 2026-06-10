import sys
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

print("Initializing model and tokenizer...")
model_name = "Hengchang-Liu/D3LM-from-nt"
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Import Config")
# Import MDMGenerationConfig from the model's auto-downloaded module
MDMGenerationConfig = getattr(sys.modules[type(model).__module__], "MDMGenerationConfig")