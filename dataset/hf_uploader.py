from datasets import load_dataset
from pathlib import Path

base = Path(__file__).resolve().parent

# point each split at its CSVs (a list of files per split works too)
structure = {
    "enzyme": {
        "mrna":    f'{base / "Enzyme" / "embl-files" / "mRna" / "csv" / "mRna.csv"}',
        "genomic": f'{base / "Enzyme" / "embl-files" / "genomic" / "csv" / "genomic.csv"}',
        "other":   f'{base / "Enzyme" / "embl-files" / "other" / "csv" / "other.csv"}',
    },
    "protein": {
        "mrna":    f'{base / "Protein" / "embl-files" / "mRna" / "csv" / "mRna.csv"}',
        "genomic": f'{base / "Protein" / "embl-files" / "genomic" / "csv" / "genomic.csv"}',
        "other":   f'{base / "Protein" / "embl-files" / "other" / "csv" / "other.csv"}',
    },
}

for config_name, data_files in structure.items():
    ds = load_dataset("csv", data_files=data_files)  # dict keys become split names
    ds.push_to_hub("DeadCowboy/Enzyme-Protein-Dataset", config_name=config_name)