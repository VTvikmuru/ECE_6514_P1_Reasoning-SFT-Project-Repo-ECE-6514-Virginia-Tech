#!/bin/bash

#-- SLURM Job Directives --#
#SBATCH --nodes=1                   # Request a single node
#SBATCH --ntasks-per-node=4         # Request 5 CPU cores
#SBATCH --time=1:00:00              # Set a 1-hour time limit
#SBATCH --partition=h200_normal_q   # Specify the GPU partition: h200_normal_q, a100_normal_q on Tinkercliffs | a30_normal_q on Falcon
#SBATCH --account=tml_2026          # Your class-specific account
#SBATCH --gres=gpu:1                # Request 1 GPU


module load Miniconda3
module load CUDA/12.8.0
conda activate lf-train-clean
conda init
source ~/.bashrc
conda activate lf-train-clean
cd workdir2
cd Proj*
cd LL*
export WANDB_PROJECT="Proj"
FORCE_TORCHRUN=1 llamafactory-cli train sft_config.yaml
