#!/bin/bash

#-- SLURM Job Directives --#
#SBATCH --nodes=1                   # Request a single node
#SBATCH --ntasks-per-node=2         # Request 2 CPU cores
#SBATCH --time=1:00:00              # Set a 1-hour time limit
#SBATCH --partition=h200_normal_q   # Specify the GPU partition: h200_normal_q, a100_normal_q on Tinkercliffs | a30_normal_q on Falcon
#SBATCH --account=tml_2026          # Your class-specific account
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --array=0-7%2		    # Run 8 benchmarks, max 2 simultaneously

module load Miniconda3
module load CUDA/12.8.0
conda activate lf-train-clean
conda init
source ~/.bashrc
conda activate lf-train-clean
cd workdir2
cd Proj*
cd eval
conda deactivate
unset SLURM_SUBMIT_DIR
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"

BENCHMARKS=(aime math cn_math_2024 kaoyan amc minerva olympiadbench gpqa)
BENCHMARK="${BENCHMARKS[$SLURM_ARRAY_TASK_ID]}"

echo "Benchmark=$BENCHMARK"

CONDA_ENV=lf-train-clean \
./eval_single.sh "$BENCHMARK"
