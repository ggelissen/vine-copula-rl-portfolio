#!/bin/bash
#SBATCH --job-name=rl_train
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --array=0-3   # 4 GPU runs

cd /gabirel/copula-based_dynamic_portfolio_selection

# Activate conda environment
source /gabirel/miniforge3/etc/profile.d/conda.sh
conda activate rl_env

# Set GPU for this array task
GPU_ID=$SLURM_ARRAY_TASK_ID
SEED=$((20260741 + GPU_ID))
RUN_NAME="full-a100-32cpu-seed-${SEED}"

# Environment variables
export TRAIN_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=$GPU_ID
export VINE_SIM_CORES=32
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export N_SIM_CVAR=10000
export PRETRAIN_EPISODES=500
export FINETUNE_EPISODES=200
export PRETRAIN_BATCH_SIZE=128
export FINETUNE_BATCH_SIZE=32
export TRAIN_SEED=$SEED
export TRAIN_OUTPUT_DIR="data/rl_runs/${RUN_NAME}"
export RETICULATE_PYTHON="$(which python)"

echo "Starting run ${RUN_NAME} on GPU ${GPU_ID} (seed ${SEED})"

# Run R script (will use cached matrices)
Rscript --vanilla rl/train_rl.r

echo "Run ${RUN_NAME} completed"