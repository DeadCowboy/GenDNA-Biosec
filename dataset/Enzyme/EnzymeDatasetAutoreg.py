import os
import json
from typing import Any
import warnings
import torch
import re
import random
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
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
        mol_type: str = "mRna",
        masking_ratio: float = 0.4,
        seq_length: int = 2048,
        tokenizer: Union[AutoTokenizer, None] = None,
        length_multiple: int = 6,
        transform=None,
        ec_table_path: Union[str, Path] = Path(__file__).resolve().parent / "enzyme-ec-table.json",
        ec_range: str = "",
    ):
        """
        Args:
            dataset_dir: Folder where the embl files are found. Should contain at least "mRna" and "genomic"
            mol_type: select type of Nucleotide Sequence. ("mRna" or "genome")
            masking_ratio (float): The percentage of nucleotides to mask. Within rounding exactly that percentage of the input will be masked if no truncation. Set to zero to deactivate masking. One for complete masking
            masking_type (diffusion, autregressive): currently does nothing
            length_multiple: The length of the sequence will be truncated to a multiple of this number. This is important for tokenizers that tokenize in n-mers
            ec_table_path: Path to the EC table JSON file. If not provided, defaults to "enzyme-ec-table.json" in the same directory as this script.
            ec_range: A string representing the prefix of EC numbers to include. So "1.2.3" would include all EC numbers starting with "1.2.3". If not provided, all EC numbers will be included.
        """
        assert mol_type in ["mRna", "genome"]
        assert masking_ratio >= 0 and masking_ratio <= 1, ValueError("Masking ratio must be between 0 and 1")
        dataset_dir = Path(__file__).resolve().parent / "embl-files"
        self.dir_path = dataset_dir / mol_type
        self.ids = [id for id in os.listdir(self.dir_path)]

        if ec_range:
            # Filter ids based on EC range
            try:
                with open(ec_table_path, "r") as f:
                    self.ec_table: Dict[str, str] = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"EC table file not found at {ec_table_path}. EC number lookups will return None.")

            filtered_ids = []
            for id in self.ids:
                with open(os.path.join(self.dir_path, id), "r") as itemFile:
                    fileContents = "".join(itemFile.readlines()[:3])
                    embl_id = self.get_embl_from_file(fileContents)
                    ec_number = self.get_ec_number(embl_id)
                    if ec_number and ec_number.startswith(ec_range):
                        filtered_ids.append(id)
            self.ids = filtered_ids

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

    def get_ec_number(self, embl_id: str | None) -> str | None:
        """
        Get the EC number for a given EMBL ID from the EC table.
        Args:
            embl_id: The EMBL ID to look up.
        Returns:
            The corresponding EC number, or None if not found.
        """
        if embl_id is None:
            return None
        try:
            return self.ec_table[embl_id]
        except KeyError:
            return None

    def get_embl_from_file(self, fileContents: str) -> str | None:
        """
        Extract the EMBL ID from the contents of an EMBL file.
        Args:
            fileContents: The contents of the EMBL file as a string.
        Returns:
        The EMBL ID if found, or None if not found.
        """
        embl_id_match = re.search(r"ID\s+(\S+);", fileContents)
        if embl_id_match:
            return embl_id_match.group(1)
        else:
            return None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        with open(os.path.join(self.dir_path, self.ids[idx]), "r") as itemFile:
            fileContents = itemFile.read()
            embl_id = self.get_embl_from_file(fileContents)
            if embl_id is None:
                warnings.warn(f"Could not find EMBL ID in file {self.ids[idx]}. EC number will be set to None.")

            sequence, metadata = self.sequenceFromFile(fileContents)
            if self.transform:
                sequence = self.transform(sequence)
            sequence = sequence.upper()  # Because the tokenizer only takes upper case nucleotides apparently

            # If sequence is too short, skip and use next sample
            if len(sequence) < self.seq_length:
                return self.__getitem__(idx + 1 if idx + 1 < len(self.ids) else random.randint(0, len(self.ids) - 1))

            # Truncate sequence, so it will for sure fit in the limited id length
            trunc_input_seq = self.truncate(sequence, length=self.seq_length)
            trunc_input_seq = self.truncate(trunc_input_seq, length=self.make_multiple_of(len(trunc_input_seq), self.length_multiple))  # Constrain to be of length multiple of 6
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
                "ec_number": self.get_ec_number(embl_id),
                "embl_id": embl_id,
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

        metadata: Dict[str, Any] = {}

        for line in lines:
            if line.startswith("SQ"):
                in_sequence = True
                continue  # Skip the line with the metainformation
            if in_sequence:
                seq = re.sub(r"[^atgc]", "", line)
                sequence_parts.append(seq)
        return "".join(sequence_parts), metadata


if __name__ == "__main__":
    dataset = SwissProtDataset(mol_type="mRna", masking_ratio=0.4, seq_length=1000)
    for idx in range(len(dataset)):
        try:
            input_ids, attention_mask, label_dict = dataset[idx]
            if label_dict["ec_number"] is None:
                print(label_dict["embl_id"])

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
