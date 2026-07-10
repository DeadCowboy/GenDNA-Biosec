import os
from typing import Any
import torch
import re
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from typing import Tuple, Dict, List, Literal, Union

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class SwissProtDataset(Dataset):
    """
    This dataset captures dna sequences of the enzyme dataset provided by SwissProt. Both mRna and genomic sequences are available
    """

    def __init__(
        self,
        dataset_dir: str = "dataset/Protein/remaped-uniprot-embl/embl-files",
        mol_type: str = "mRna",
        masking_ratio: float = 0.4,
        seq_length: int = 2048,
        tokenizer: Union[AutoTokenizer, Literal["D3LM", "Carbon"]] = "Carbon",
        length_multiple: int = 6,
        transform=None,
    ):
        """
        Args:
            dataset_dir: Folder where the embl files are found. Should contain at least "mRna" and "genomic"
            mol_type: select type of Nucleotide Sequence. ("mRna" or "genome")
            masking_ratio (float): The percentage of nucleotides to mask. Within rounding exactly that percentage of the input will be masked if no truncation. Set to zero to deactivate masking. One for complete masking
            masking_type (diffusion, autregressive): currently does nothing
            length_multiple: The length of the sequence will be truncated to a multiple of this number. This is important for tokenizers that tokenize in n-mers
        """
        assert mol_type in ["mRna", "genome"]
        assert masking_ratio >= 0 and masking_ratio <= 1, ValueError("Masking ratio must be between 0 and 1")
        self.dir_path = os.path.join("/user", "laurenz.kammeyer", "u26834", "GenDNA_Biosec", dataset_dir, mol_type)
        self.ids = [id for id in os.listdir(self.dir_path)]
        if type(tokenizer) == AutoTokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceBio/Carbon-3B", trust_remote_code=True)
        self.masking_ratio = masking_ratio
        self.seq_length = seq_length
        self.length_multiple = length_multiple
        self.transform = transform
        self.cuda_avail = torch.cuda.is_available()

    def __len__(self):
        return len(self.ids)

    def truncate(self, sequence, length: int):
        return sequence[:length]

    def make_multiple_of(self, number: float, multiple: int) -> int:
        """
        Make a number a multiple of another number by rounding down
        """
        return int(number // multiple) * multiple

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        with open(os.path.join(self.dir_path, self.ids[idx]), "r") as itemFile:
            fileContents = itemFile.read()
            sequence = self.sequenceFromFile(fileContents)
            if self.transform:
                sequence = self.transform(sequence)
            sequence = sequence.upper()  # Because the tokenizer only takes upper case nucleotides apparently

            # Truncate sequence, so it will for sure fit in the limited id length
            trunc_input_seq = self.truncate(sequence, length=self.seq_length)
            trunc_input_seq = self.truncate(sequence, length=self.make_multiple_of(len(sequence), self.length_multiple))  # Constrain to be of length multiple of 6
            assert len(trunc_input_seq) % self.length_multiple == 0, ValueError(f"Truncated sequence length {len(trunc_input_seq)} is not a multiple of {self.length_multiple}")

            # Add DNA Prompt (just how the model was trained)
            prompt = f"<dna>{trunc_input_seq}"

            out = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            )  # pyright: ignore[reportCallIssue] # {'input_ids': [3,...], 'attention_mask': [1111...111]}   # 3 is the cls token
            if self.cuda_avail:
                out = out.to("cuda")

            input_ids = out["input_ids"]
            attention_mask = out["attention_mask"]
            target_length = int(input_ids.numel() * (1 - self.masking_ratio))
            input_ids = input_ids[:, :target_length]  # Masking Ratio of 1 should be full masking
            attention_mask = attention_mask[:, :target_length]
            label_dict = {
                "overlength_original_sequence": sequence,
                "fulllength_ids": out["input_ids"],
                "input_sequence": trunc_input_seq,
                "fulllength_attention_mask": out["attention_mask"],
                "target_length": target_length,
            }

            return input_ids, attention_mask, label_dict

    def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """
        Args:
            batch: list of sequences (lists or 1D tensors of token ids)

        Returns:
            padded_batch: LongTensor of shape (batch_size, max_len)
            sequences are cut to the length of the shortest input
        """

        min_length = np.min([len(seq) for seq, _, _ in batch])
        # cut_batch = [(seq[:min_length], att_mask[:min_length], label) for seq, att_mask, label in batch]
        # # Stack sequence and attention mask tensors
        # cut_batch = [(torch.stack((seq, att_mask)), label) for seq, att_mask, label in cut_batch]
        input_ids = torch.cat([seq[:min_length] for seq, _, _ in batch], dim=0)
        attention_masks = torch.cat([att_mask[:min_length] for _, att_mask, _ in batch], dim=0)
        label_batch = [label for _, _, label in batch]
        return input_ids, attention_masks, label_batch

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
