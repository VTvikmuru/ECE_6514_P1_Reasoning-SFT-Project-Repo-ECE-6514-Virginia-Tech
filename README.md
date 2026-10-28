# Reasoning SFT Project Repo | ECE 6514, Virginia Tech

This repository contains the required files for  Project 1, "SFT LLM for Enhancing Reasoning", for ECE 6514 at Virginia Tech. The project focuses on choosing effective training data and using SFT to optimize an LLM.

This project runs on the VT ARC system. Important and required files have been exported to this repository in a file structure that mirrors that on ARC.

## Project layout

The project files are under `Home/`, corresponding to the `~` home directory on ARC.

| Folder/File | Description |
| --- | --- |
| `run_training_new.sh` | Script for training model on random 15k samples |
| `run_training_new_advanced.sh` | Script for training model on k-means 15k samples (K=15) |
| `run_training_new_advanced_centroid.sh` | Script for training model on k-means centroid 15k samples (K=300) |
| `run_training_new.sh` | Script to fine tune model with 15k random (pre-sampled) samples |
| `run_training_new_advanced.sh` | Script to fine tune model with 15K samples pre-selected through K-means |
| `run_training_new_advanced_centroid.sh` | Script to fine tune model with 15K samples pre-selected through K-means and centroid |
| `run_eval_base.sh` | Script to evaluate model without fine any tuning on all benchmarks |
| `run_eval_random.sh` | Script to evaluate model fine tuned with random samples on all benchmarks |
| `run_eval_advanced.sh` | Script to evaluate model fine tuned with k-means samples on all benchmarks |
| `workdir2/Advanced-Dataset-Creation/` | Source data, K-means and K-means centroid sampling scripts, and generated datasets |
| `workdir2/Project-Reasoning-SFT-LLM/LLaMA-Factory/` | LLama Factory config files (yaml), training datasets, and modified name-file mapping file |
| `workdir2/Project-Reasoning-SFT-LLM/eval/` | Evaluation output logs for baseline mode, model fine tuned on random samples, and model fine tuned on K-means samples |
| `requirements.txt` | Conda environment setup requirements file|

## Environment Setup

All files in this repository are intended to run on Virginia Tech ARC clusters, with access to a h200 or a100 GPU. 
They require the LLaMA factory files to facilitate the fine tuning of the model.

Clone the LLaMA Factory using the commands below:
```bash
cd ~
cd workdir2
git clone --depth 1  https://github.com/reds-lab/Project-Reasoning-SFT-LLM 
cd LLaMA-Factory
```
The repo `https://github.com/reds-lab/Project-Reasoning-SFT-LLM` contains more detailed insutrctions. 

The included requirements.txt file can be used to set up the Conda virtual environment with all required dependencies in their required versions. For the scripts in this repository to work, the environment must be called `lf-train-clean`.

Use the following commands to create the environment:

```bash
module load Miniconda3
module load CUDA/12.6.0
cd ~
conda create --name lf-train-clean --file requirements.txt
```

To activate the environment, use:

```bash
conda activate lf-train-clean
```

To dactivate the enviornment use:
```bash
conda deactivate
```

The Huggingface cli should be used to download the `acereason11_100k.json` dataset file from the repo ID `redsgnaoh/acereason11_100k`.

## Training

First, place the `acereason11_100k.json` dataset file in the `workdir2\Advanced-Dataset-Creation` directory. Then run the following commands to generate the random dataset called `acereason11_15k_sample.json`, the k-means dataset called `advancedkMeans_15k_sampled.json`, and the k-means centroid dataset called `advancedkMeans_15k_sampled_centroid.json`:

```bash
python jsonRandomSample.py
python advancedSampleGeneratorKMeans.py
python advancedSampleGeneratorKMeans_centroid.py
python cleanData.py
python cleanDataCentroid.py
```

Copy the generated files into the folder `workdir2\Project-Reasoning-SFT-LLM\LLaMA-Factory\data`.

Finally, training can be started by executing the following comamnds (from the repo root directory) corresponding to the type of training desired (described in project layout):
```bash
sbatch run_training_new.sh
sbatch run_training_new_advanced.sh
sbatch run_training_new_advanced_centroid.sh
```

## Evaluation

Similar to the commands for training, the following commands can be used to perform evaluation from the repo root directory:
```bash
sbatch run_eval_base.sh
sbatch run_eval_random.sh
sbatch run_eval_advanced.sh
```

The evaluation results are saved into seperate folders for each script, in the directory `\workdir2\Project-Reasoning-SFT-LLM\eval`


Note: Full names have been replaced for privacy on this public repo.