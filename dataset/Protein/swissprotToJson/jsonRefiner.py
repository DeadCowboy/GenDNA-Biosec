import json
from pathlib import Path

DEFAULT_INPUT = Path("/home/lkammeyer/Documents/Uni/Master_Thesis/GenDNA-Biosec/carbon/dataset/Protein/swissprotToJson/swissProtEnzyme.json")


def load_swissprot_enzyme(file_path=DEFAULT_INPUT):
    """Load swissProtEnzyme.json and return the parsed Python object."""
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    data = load_swissprot_enzyme()

    # Output to CSV
    with open("swissProtEnzyme_sizes.csv", "w", encoding="utf-8") as handle:
        for key, value in data.items():
            handle.write(f"{key},{len(value)}\n")
