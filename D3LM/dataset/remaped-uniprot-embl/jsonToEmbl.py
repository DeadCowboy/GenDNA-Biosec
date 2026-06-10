import requests, sys, json, re
from tqdm import tqdm
from time import sleep
from typing import List, Dict


baseDir = "embl_alt_dir"

def extractID(fileString: str) -> str:
  match = re.search(r"ID\s+([^;]+)", fileString)
  if match:
    text : str = match.group(1)
    return text
  else:
    return "Match did not work!"

def removeP1(id: str) -> str:
  if id[-2] == ".":
    return id[:-2]
  else:
    return id

def batchQueryENA(swissProtIDs: List[str]) -> str:
  assert len(swissProtIDs) <= 10000, "Max length of 10000 accessions per request"
  body = {
  "accessions": swissProtIDs,
  "expanded": "true",
  "annotationOnly": "false",
  "lineLimit": 0,
  "download": "false",
  "gzip": "false",
  "set": "true",
  "includeLinks": "true",
  "range": "string",
  "complement": "true"
  }
  headers = {
    "accept": "text/plain",
    "Content-Type": "application/json"
  }
  base_url = "https://www.ebi.ac.uk/ena/browser/api/embl"

  response = requests.post(base_url, headers=headers, json=body)
  if not response.ok:
    print(response.text)
    response.raise_for_status()
    # sys.exit()

  return response.text

def isMRna(fileString: str):
  match = re.search('mol_type="([^"]*)"', fileString)
  if match:
    capture = match.group(1)
    if capture == "mRNA":
      return True
    elif capture in ["genomic DNA","genomic RNA", "transcribed RNA"]:
      return False
    else:
      print(f"Warning. Could not determine if file is genomic or mRna. Unknown String. Regex extracted: '{capture}' ")
      return None
  
  else:
    print(f"Warning. Could not determine if file is genomic or mRna. Unknown String. File String: /n /n {fileString} /n /n{'-'*30} EOF")
    return None
# /mol_type="genomic DNA"

with open("combined-remap.json",'r') as mapJsonFile:
  mapJsonStr = mapJsonFile.read()
  mapJson: Dict[str,List[str]] = json.loads(mapJsonStr)  

  batch = []
  def queryBatch():
    rawQueryResponse = batchQueryENA(batch)
    # emblFileStrings = rawQueryResponse.split("//")[:-1]
    emblFileStrings = re.split("(?<!:)//", rawQueryResponse)[:-1]

    
    if len(batch) != len(emblFileStrings): print(f"Requested {len(batch)} files, but only got {len(emblFileStrings)} files.\
                                                  Probably Broken Link for {list(set([removeP1(elem) for elem in batch]).difference([extractID(elem) for elem in emblFileStrings]))}")
    for enaID, fileString in zip(batch, emblFileStrings):
      type = isMRna(fileString)
      if type == True:
        with open(f"{baseDir}/mRna/{enaID}.embl",'w') as currentFile:
          currentFile.write(fileString)
      elif type == False:
        with open(f"{baseDir}/genomic/{enaID}.embl",'w') as currentFile:
          currentFile.write(fileString)
      else:
        with open(f"{baseDir}/other/{enaID}.embl",'w') as currentFile:
          currentFile.write(fileString)



  for idList in tqdm(mapJson.values()):
    for id in idList:
      batch.append(id)

      if len(batch) == 1000:
        queryBatch()
        sleep(1) # Sleep for 1 seconds to not overload the server
        batch = []

  queryBatch()
print("Done")



# cd D3LM/dataset_creation/remaped-uniprot-embl/