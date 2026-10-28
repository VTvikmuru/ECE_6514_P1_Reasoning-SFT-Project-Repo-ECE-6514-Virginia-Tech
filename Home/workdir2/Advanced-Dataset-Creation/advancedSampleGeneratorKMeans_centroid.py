import os

# Limit threads to prevent crash
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

INPUT_FILE = "acereason11_100k.json"
OUTPUT_FILE = "advancedkMeans_15k_sampled_centroid.json"

EMBEDDINGS_FILE = "embeddings.npy"

K = 300
TOTAL_SAMPLES = 15000
SAMPLES_PER_CLUSTER = TOTAL_SAMPLES // K

# Extract instruction text
print("1) Loading data from JSON...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

instructions = [item.get("instruction", "") for item in data]


if os.path.exists(EMBEDDINGS_FILE):
	embeddings = np.load(EMBEDDINGS_FILE)
else:
	print("2) Generating embeddings...")
	model = SentenceTransformer("all-MiniLM-L6-v2") # Faster embedding model
	embeddings = model.encode(instructions, batch_size=64, show_progress_bar=True)

	# Save to avoid having to run embedding again
	np.save(EMBEDDINGS_FILE, embeddings)

print("3) Performing clustering...")
kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto")
clusterLabels = kmeans.fit_predict(embeddings)

print("4) Group and sample randomly...")
#clusterIndices = {i: [] for i in range(K)} ## Create list
#for idx, label in enumerate(clusterLabels):
#    clusterIndices[label].append(idx)

#sampledItems = []
#for clusterID, indices in clusterIndices.items():
#	numToSample = min(len(indices), SAMPLES_PER_CLUSTER) # In case cluster has lower items than expected
#
#	chosenIndices = random.sample(indices, numToSample)
#
#	for i in chosenIndices:
#	        sampledItems.append(data[i])
#
#	print("Cluster ID: ", clusterID, " Items: ", len(indices), " Samples: ", numToSample)

distances = cdist(embeddings,kmeans.cluster_centers_, metric='euclidean')

sampledItems=[]
for clusterID in range(K):
	mask = (clusterLabels == clusterID)
	clusterDists = distances[:, clusterID]

	sortedIndexes = np.argsort(clusterDists)
	assignedIndexes = [i for i in sortedIndexes if mask[i]]

	numToSample = min(len(assignedIndexes), SAMPLES_PER_CLUSTER) # In case cluster has lower items than expected
	selectedIndexes = assignedIndexes[:numToSample]

	for i in selectedIndexes:
		sampledItems.append(data[i])

if len(sampledItems) < TOTAL_SAMPLES:
	missing = TOTAL_SAMPLES - len(sampledItems)
	sampledItems.extend(random.choices(sampledItems, k=missing))


print("Saving to file")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
	json.dump(sampledItems, f, ensure_ascii=False, indent=2)


