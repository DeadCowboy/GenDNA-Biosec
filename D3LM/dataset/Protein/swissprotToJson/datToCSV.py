import re
from collections import Counter

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
        if line.startswith("DR"):
            lineWithoutDR = line[2:]
            longAccessions = lineWithoutDR.split(";")[:-1]
            for uncleanAccession in longAccessions:
                cleanString = uncleanAccession.split(",")[0].strip()
                accession_ids.append(cleanString)


        # print(result)
 
counter = Counter(accession_ids)
print( [item for item, count in counter.items() if count > 1])
 
# Split up for 100k entity limit
# File has 238k
""" with open("swissProtEnzyme1.csv", "w") as file:    
    # Write the accession_ids in a csv
    for id in accession_ids[0:90000]:
        file.write(id + ",")

with open("swissProtEnzyme2.csv", "w") as file:    
    # Write the accession_ids in a csv
    for id in accession_ids[90000:180000]:
        file.write(id + ",")

with open("swissProtEnzyme3.csv", "w") as file:    
    # Write the accession_ids in a csv
    for id in accession_ids[180000:-1]:
        file.write(id + ",") """


