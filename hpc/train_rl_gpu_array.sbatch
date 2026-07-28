#!/usr/bin/env bash
# Submit from the repository root with:
#   sbatch hpc/train_rl_gpu_array.sbatch
# One array task = one independent seed = one exclusive GPU.
# Adjust the partition/account/module lines to the local cluster.
#SBATCH --job-name=vine-rl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/vine-rl-%A_%a.out
#SBATCH --error=logs/vine-rl-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root.}"
mkdir -p logs "data/rl_runs"

# Site-specific: uncomment/adapt these when the cluster uses environment modules.
# module purge
# module load R/4.4 CUDA/12.4

# Activate the one conda environment that reticulate will use.  Set
# RL_CONDA_ENV before submission, e.g. export RL_CONDA_ENV=vine-rl.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RL_CONDA_ENV:?Set RL_CONDA_ENV to the conda environment name.}"

# Slurm sets CUDA_VISIBLE_DEVICES to this task's allocated GPU.  Leave it intact.
export TRAIN_DEVICE=cuda
export TRAIN_SEED="$((20260727 + SLURM_ARRAY_TASK_ID))"
export TRAIN_OUTPUT_DIR="data/rl_runs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export RETICULATE_PYTHON="${CONDA_PREFIX}/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

nvidia-smi
Rscript --vanilla rl/train_rl.r
