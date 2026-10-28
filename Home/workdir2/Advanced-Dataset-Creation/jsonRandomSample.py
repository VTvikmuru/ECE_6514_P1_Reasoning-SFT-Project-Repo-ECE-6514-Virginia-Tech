import json
import random

filename = "acereason11_100k.json"
outputFilename = "acereason11_15k_sample.json"
datasetSize = 15000

with open(filename, "r", encoding="utf-8") as f:
  all_items = json.load(f)

#  15k random items
sampled_items = random.sample(all_items, datasetSize)

#  Remove unnecessary fields
cleanedItems = []
for item in sampled_items:
  firstFourKeys = list(item.keys())[:4]
  item = {k: item[k] for k in firstFourKeys}
  cleanedItems.append(item)

# Write output file
with open(outputFilename, "w", encoding="utf-8") as f:
  json.dump(cleanedItems, f, indent=4, ensure_ascii=False)
