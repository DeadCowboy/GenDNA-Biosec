import json

# open three files

with open("Enzyme1-remap.json", "r") as file1, open("Enzyme2-remap.json", "r") as file2, open("Enzyme3-remap.json", "r") as file3:
    shard1 = json.load(file1)
    shard2 = json.load(file2)
    shard3 = json.load(file3)
    combinedList = shard1["results"] + shard2["results"] + shard3["results"]
    print(len(combinedList))
    print(len(shard1["results"]), len(shard2["results"]), len(shard3["results"]))

    combinedDict = {}
    for item in combinedList:
        combinedDict[item["from"]] = []

    for item in combinedList:
        combinedDict[item["from"]].append(item["to"])

    with open("combined-remap.json", "w") as outfile:
        json.dump(combinedDict, outfile)
