import os
from typing import Any
import torch
import re
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from typing import Tuple, Dict

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class SwissProtDataset(Dataset):

    def __init__(self, dataset_dir: str = "dataset/remaped-uniprot-embl/embl-files", mol_type: str = "mRna", masking_ratio: float = 0.4, max_length: int = 2048, transform=None):
        """
        Args:
            dataset_dir: Folder where the embl files are found. Should contain at least "mRna" and "genomic"
            mol_type: select type of Nucleotide Sequence. ("mRna" or "genome")
            masking_ratio (float): The percentage of nucleotides to mask. Within rounding exactly that percentage of the input will be masked if no truncation. Set to zero to deactivate masking. One for complete masking
        """
        assert mol_type in ["mRna", "genome"]
        assert masking_ratio >= 0 and masking_ratio <= 1, ValueError("Masking ratio must be between 0 and 1")
        self.dir_path = os.path.join("/user", "laurenz.kammeyer", "u26834","GenDNA_Biosec" ,"D3LM", dataset_dir, mol_type)
        self.ids = [id for id in os.listdir(self.dir_path)]
        self.model_name = "Hengchang-Liu/D3LM-from-nt"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.transform = transform
        self.masking_ratio = masking_ratio
        self.max_length = max_length
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        with open(os.path.join(self.dir_path, self.ids[idx]), "r") as itemFile:
            fileContents = itemFile.read()
            sequence = self.sequenceFromFile(fileContents)
            if self.transform:
                sequence = self.transform(sequence)
            sequence = sequence.upper()  # Because the tokenizer only takes upper case nucleotides apparently
            out = self.tokenizer(sequence)  # {'input_ids': [3,...], 'attention_mask': [1111...111]}   # 3 is the cls token
            input_ids = out["input_ids"]
            attention_mask = out["attention_mask"]
            trunc_input_ids = self.truncate(input_ids, self.max_length)

            # Mask input ids directly
            masked_input_ids, masked_indices = self.mask(trunc_input_ids, masking_ratio=self.masking_ratio, mask_token=self.mask_token_id)
            masked_input_ids = torch.tensor(masked_input_ids)
            attention_mask = torch.tensor(attention_mask)

            label_dict = {"original_sequence": sequence,        "orig_input_ids":   out["input_ids"],
                          "trunc_input_ids":   trunc_input_ids, "masked_input_ids": masked_input_ids,
                          "masked_indices":    masked_indices}
            return masked_input_ids, label_dict

    def collate_fn(self, batch):
        """
        Args:
            batch: list of sequences (lists or 1D tensors of token ids)
            pad_token_id: token used for padding

        Returns:
            padded_batch: LongTensor of shape (batch_size, max_len)
        """

        masked_ids = []
        label_batch = []
        for masked, label in batch:
            masked_ids.append(torch.as_tensor(masked, dtype=torch.long))
            label_batch.append(label)

        padded_batch = pad_sequence(masked_ids, batch_first=True, padding_value=self.pad_token_id)
        return padded_batch, label_batch
    
    def truncate(self,sequence, length:int):
        return sequence[:length]

    def mask(self, sequence, masking_ratio:float, mask_token):
        # Replace proportion of indexes, chosen randomly, with the mask token

        # Start from 1 to exclude the classification token as the first token
        random_indices = random.sample(range(1,len(sequence)), int(len(sequence) * self.masking_ratio))  # Sample without replacement
        masked_bp = sequence.copy()
        for idx in random_indices:
            masked_bp[idx] = self.mask_token_id
        if type(sequence) == str:
            print("Was a string")
            return "".join(masked_bp)
        return masked_bp, random_indices

    def sequenceFromFile(self, itemFile: str):
        # Extract the Basepair sequence from an embl file string
        lines = itemFile.splitlines()
        in_sequence = False
        sequence_parts = []

        for line in lines:
            if line.startswith("SQ"):
                in_sequence = True
                continue  # Skip the line with the metainformation
            if in_sequence:
                seq = re.sub(r"[^atgc]", "", line)
                sequence_parts.append(seq)
        return "".join(sequence_parts)


# uut = SwissProtDataset()

# with open("embl-files/mRna/AAA02563.1.embl") as inputFile:
#     test = inputFile.read()
#     print(uut.__getitem__(1))
