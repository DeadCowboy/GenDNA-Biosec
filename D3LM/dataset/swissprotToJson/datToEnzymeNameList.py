import json
import re

# Example usage
file_path = 'enzyme.dat'  # Replace with your input file path
output_json_path = 'swissProtEnzyme.json'  # Replace with your desired output JSON file path
result = {}

ec_id_pattern = re.compile(r'^ID\s+(.*?)\s*$')

current_ec_id = None
accession_ids = []
count = 0
with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if line.startswith("ID"):
            if current_ec_id is not None:
                if accession_ids:
                    result[current_ec_id] = accession_ids
                    accession_ids = []
            match = ec_id_pattern.match(line)
            assert match
            current_ec_id = match.group(1)

        if line.startswith("DR"):
            lineWithoutDR = line[2:]
            longAccessions = lineWithoutDR.split(";")[:-1]
            for uncleanAccession in longAccessions:
                cleanString = uncleanAccession.split(",")[0].strip()
                accession_ids.append(cleanString)


        # print(result)
with open(output_json_path, "w") as json_file:
    json.dump(result, json_file, indent=4)

