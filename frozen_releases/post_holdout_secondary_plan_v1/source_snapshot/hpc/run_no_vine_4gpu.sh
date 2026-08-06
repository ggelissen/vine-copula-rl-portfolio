#!/usr/bin/env bash
# Run the frozen ten-seed no-vine secondary experiment on four independent GPUs.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd -P)}"
RSCRIPT="${RSCRIPT:-Rscript}"
PYTHON="${PYTHON:-python3}"
: "${TRAIN_PYTHON:?Set TRAIN_PYTHON to the GPU-enabled Python used by reticulate for training}"
SWEEP_ROOT="${NO_VINE_SWEEP_ROOT:-$REPO_ROOT/data/no_vine_rl_runs_secondary_v1}"
CORES_PER_WORKER="${VINE_SIM_CORES_PER_WORKER:-18}"
WORKERS=4
CONTRACT="$REPO_ROOT/publication_pipeline_draft/config/secondary_experiments_v1.json"

cd "$REPO_ROOT"
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC

if [[ ! "$CORES_PER_WORKER" =~ ^[1-9][0-9]*$ ]]; then
  echo "VINE_SIM_CORES_PER_WORKER must be a positive integer." >&2
  exit 2
fi
if [[ ! -x "$TRAIN_PYTHON" ]]; then
  echo "TRAIN_PYTHON is missing or not executable: $TRAIN_PYTHON" >&2
  exit 2
fi
"$TRAIN_PYTHON" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("TRAIN_PYTHON cannot see CUDA; refusing a silent CPU sweep.")
if torch.cuda.device_count() < 4:
    raise SystemExit(
        f"Four GPUs are required by this launcher; PyTorch sees {torch.cuda.device_count()}."
    )
print(
    "Training runtime:", sys.executable,
    "| torch", torch.__version__,
    "| CUDA", torch.version.cuda,
    "| GPUs", torch.cuda.device_count(),
)
PY
if [[ -e "$SWEEP_ROOT" ]]; then
  echo "Refusing to mix or overwrite a prior sweep: $SWEEP_ROOT" >&2
  echo "Set NO_VINE_SWEEP_ROOT to a new empty path for a new declared run." >&2
  exit 2
fi

mkdir -p "$SWEEP_ROOT/worker_logs"
declare -a pids=()

for worker in $(seq 1 "$WORKERS"); do
  gpu=$((worker - 1))
  status="$SWEEP_ROOT/worker_status_${worker}.csv"
  log="$SWEEP_ROOT/worker_logs/worker_${worker}.log"
  echo "Launching worker $worker/$WORKERS on CUDA device $gpu"
  env -u CONDA_PREFIX \
    CUDA_VISIBLE_DEVICES="$gpu" \
    RETICULATE_PYTHON="$TRAIN_PYTHON" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VINE_SIM_CORES="$CORES_PER_WORKER" \
    SWEEP_SEEDS_FILE="config/no_vine_ablation_seeds.yaml" \
    SWEEP_ROOT_DIR="$SWEEP_ROOT" \
    SWEEP_WORKER_COUNT="$WORKERS" \
    SWEEP_WORKER_INDEX="$worker" \
    SWEEP_STATUS_FILE="$status" \
    VINE_OBSERVATION_MODE=zero \
    "$RSCRIPT" --vanilla rl/run_seed_sweep.r config/config.yaml \
    >"$log" 2>&1 &
  pids+=("$!")
done

failed=0
for worker in $(seq 1 "$WORKERS"); do
  if ! wait "${pids[$((worker - 1))]}"; then
    echo "Worker $worker failed; see $SWEEP_ROOT/worker_logs/worker_${worker}.log" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one worker failed. Statuses/logs are preserved; do not aggregate." >&2
  exit 1
fi

"$PYTHON" publication_pipeline_draft/secondary_experiment_protocol.py \
  merge-sweep-status \
  --contract "$CONTRACT" \
  --experiment no_vine_td3 \
  --inputs \
    "$SWEEP_ROOT/worker_status_1.csv" \
    "$SWEEP_ROOT/worker_status_2.csv" \
    "$SWEEP_ROOT/worker_status_3.csv" \
    "$SWEEP_ROOT/worker_status_4.csv" \
  --output "$SWEEP_ROOT/seed_sweep_status.csv"

echo "All ten no-vine seeds and gates passed: $SWEEP_ROOT/seed_sweep_status.csv"
