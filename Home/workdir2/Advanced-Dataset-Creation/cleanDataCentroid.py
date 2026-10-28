import json

INPUT_FILE = "advancedkMeans_15k_sampled_centroid.json"
OUTPUT_FILE = "advancedkMeans_15k_sampled_centroid_cleaned.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
	data = json.load(f)

cleanedData = []
for item in data:
	keys = list((item.keys()))

	newItem = {}
	for i , key in enumerate(keys):
		if i < 4:
			newItem[key] = item[key]
	cleanedData.append(newItem)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
	json.dump(cleanedData, f, ensure_ascii=False, indent=2)
