from typing import List
import json
from pathlib import Path
import os

with open(Path("dataset/Protein/swissprotToJson/swissProtEnzyme.json"), "r") as f:
    swissProtData = json.load(f)

with open(Path("dataset/Protein/remaped-uniprot-embl/combined-remap.json"), "r") as f:
    toEmbl = json.load(f)

# Create a mapping from SwissProt IDs to EC numbers
swissProtToEC = {}
for EC, swissId in swissProtData.items():
    for id in swissId:
        swissProtToEC[id] = EC

# Sanity check
for key, value in swissProtToEC.items():
    if type(key) == list or type(value) == list:
        raise ValueError(f"Key is a list: {key}")


# Transform a swiss ID into a list of all corresponding EMBL IDs
notFound = []


def swissToEmbl(swissProtId) -> List[str]:
    def removeDotN(emblId: str) -> str:
        """
        Remove the .1 or .2 ... suffix from an EMBL ID if it exists.
        Args:
            emblId: The EMBL ID to process.
        """
        if "." in emblId:
            return emblId.split(".")[0]
        else:
            return emblId

    try:
        emblIdList: List[str] = toEmbl[swissProtId]
    except:
        global notFound
        notFound.append(swissProtId)
        return []

    assert emblIdList is not None, f"Could not find EMBL ID for SwissProt ID: {swissProtId}"
    assert type(emblIdList) == list, "There was actually an item in the embl list. Crazy"

    return [removeDotN(emblId) for emblId in emblIdList]


# Make a mapping of embl ids to EC numbers
emblToEC = {}
for swissID, ECNumber in swissProtToEC.items():
    for emblId in swissToEmbl(swissID):
        emblToEC[emblId] = ECNumber
print(f"Failed to find EMBL IDs for {len(notFound)} SwissProt IDs")

for key, value in list(emblToEC.items())[0:5]:
    print(f"{key}: {value}")


output_path = Path(__file__).with_name("enzyme-ec-table.json")
with open(output_path, "w") as f:
    json.dump(emblToEC, f, indent=2, sort_keys=True)

print(f"Saved emblToEC to: {output_path}")
