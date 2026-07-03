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
        model_name: Literal["D3LM", "Carbon"] = "Carbon",
        do_masking: bool = True,
        masking_ratio: float = 0.4,
        max_length: int = 2048,
        masking_type: Literal["diffusion", "autoregressive"] = "autoregressive",
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
        self.model_name = model_name
        if type(tokenizer) == AutoTokenizer:
            self.tokenizer = tokenizer
        elif self.model_name == "D3LM":
            self.tokenizer = AutoTokenizer.from_pretrained("Hengchang-Liu/D3LM-from-nt", trust_remote_code=True)
            self.pad_token: str = self.tokenizer.pad_token
            self.pad_token_id: int = self.tokenizer.pad_token_id
            self.mask_token: str = self.tokenizer.mask_token
            self.mask_token_id: int = self.tokenizer.mask_token_id
        elif self.model_name == "Carbon":
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceBio/Carbon-3B", trust_remote_code=True)
        else:
            raise ValueError("Tokenizer must be either an AutoTokenizer or one of the strings 'D3LM' or 'Carbon'")
        self.do_masking = do_masking
        self.masking_ratio = masking_ratio
        self.masking_type = masking_type
        self.max_length = max_length
        self.length_multiple = length_multiple
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        with open(os.path.join(self.dir_path, self.ids[idx]), "r") as itemFile:
            fileContents = itemFile.read()
            sequence = self.sequenceFromFile(fileContents)
            if self.transform:
                sequence = self.transform(sequence)
            sequence = sequence.upper()  # Because the tokenizer only takes upper case nucleotides apparently

            # Truncate sequence, so it will for sure fit in the limited id length
            trunc_input_seq = self.truncate(sequence, length=self.max_length)

            # Constrain to be of length multiple of 6
            trunc_input_seq = trunc_input_seq[: (len(trunc_input_seq) // self.length_multiple) * self.length_multiple]

            out = self.tokenizer(trunc_input_seq)  # {'input_ids': [3,...], 'attention_mask': [1111...111]}   # 3 is the cls token
            input_ids = out["input_ids"]
            attention_mask = out["attention_mask"]

            # attention_mask = [0] * len(attention_mask)

            # Mask input ids directly
            # if self.masking_type == "diffusion":
            #     masked_input_ids, masked_indices = self.mask(input_ids, masking_ratio=self.masking_ratio, mask_token=self.mask_token_id)
            # elif self.masking_type == "autoregressive":
            #     masked_input_ids, masked_indices = self.maskAutoreg(input_ids, masking_ratio=self.masking_ratio, mask_token=self.mask_token_id)
            if self.do_masking:
                input_ids, masked_indices = self.mask(input_ids, masking_ratio=self.masking_ratio, mask_token=self.mask_token_id)

            # Padding seems to not affect the model
            # padded_input_ids = self.pad(masked_input_ids, self.max_length)
            padded_input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

            padded_input_ids = torch.stack((padded_input_ids, attention_mask))

            label_dict = {"original_sequence": sequence, "orig_input_ids": out["input_ids"], "trunc_input": trunc_input_seq, "input_ids": input_ids}
            if self.do_masking:
                label_dict["masked_indices"] = masked_indices
            return padded_input_ids, label_dict

    def collate_fn(self, batch: List[Tuple[List[int] | str, Dict[str, Any]]]) -> Tuple[str | List[int], Dict[str, Any]]:
        """
        Args:
            batch: list of sequences (lists or 1D tensors of token ids)

        Returns:
            padded_batch: LongTensor of shape (batch_size, max_len)

            In case of D3LM, sequence is extended with tokenizer pad token
            In caes of Carbon, sequences are cut to the length of the shortest input
        """
        if self.model_name == "D3LM":
            masked_ids = []
            label_batch = []
            for masked, label in batch:
                masked_ids.append(torch.as_tensor(masked, dtype=torch.long))
                label_batch.append(label)

            padded_batch = pad_sequence(masked_ids, batch_first=True, padding_value=self.pad_token_id)
            return padded_batch, label_batch

        elif self.model_name == "Carbon":
            min_length = np.min([len(seq) for seq, _ in batch])
            cut_batch = [(seq[:min_length], label) for seq, label in batch]
            input_batch, label_batch = zip(*cut_batch)
            return input_batch, label_batch

    def truncate(self, sequence, length: int):
        return sequence[:length]

    def mask(self, sequence: List[str | int], masking_ratio: float, mask_token):
        # Replace proportion of indexes, chosen randomly, with the mask token

        # Start from 1 to exclude the classification token as the first token
        random_indices = random.sample(range(1, len(sequence)), int(len(sequence) * self.masking_ratio))  # Sample without replacement
        masked_bp = sequence.copy()
        for idx in random_indices:
            masked_bp[idx] = self.mask_token_id
        if type(sequence) == str:
            print("Was a string")
            return "".join(masked_bp)
        return masked_bp, random_indices

    # Version with mask_token
    # def maskAutoreg(self, sequence: List[str | int], masking_ratio: float, mask_token):
    #     non_masked_part = sequence[: np.floor(len(sequence) * (1 - masking_ratio))]
    #     masked_part = [mask_token] * np.ceil(len(sequence) * (1 - masking_ratio))
    #     masked_sequence = non_masked_part + masked_part
    #     assert len(masked_sequence) == len(sequence), "Your Math doesnt add up boi"
    #     masked_indices = list(range(np.floor(len(sequence) * (1 - masking_ratio)), len(sequence)))
    #     return masked_sequence, masked_indices

    def pad(self, input_ids: List[str | int], length: int, pad_token: str | int = None):  # type: ignore
        """
        Pad a sequence to length with pad_token. Default pad token is tokenizer.pad_token_id or tokenizer.pad_token if list of string is given
        """
        if not pad_token:
            if type(input_ids[0]) == str:
                pad_token = self.pad_token
            else:
                pad_token = self.pad_token_id

        if len(input_ids) > length:
            print("Padding target length is shorter than sequence itself! Returning input sequence")
            return input_ids

        if len(input_ids) == length:
            return input_ids

        pad_list = [pad_token] * length
        pad_list[: len(input_ids)] = input_ids
        return pad_list

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
