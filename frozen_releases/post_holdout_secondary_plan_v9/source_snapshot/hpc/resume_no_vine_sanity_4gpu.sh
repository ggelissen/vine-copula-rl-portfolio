#!/usr/bin/env bash
# Resume only the sanity gates after validated mode-marker recovery.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd -P)}"
RSCRIPT="${RSCRIPT:-Rscript}"
PYTHON="${PYTHON:-python3}"
: "${TRAIN_PYTHON:?Set TRAIN_PYTHON to the GPU training interpreter}"
SWEEP_ROOT="${NO_VINE_SWEEP_ROOT:-$REPO_ROOT/data/no_vine_rl_runs_secondary_v2}"
WORKERS=4
CONTRACT="$REPO_ROOT/publication_pipeline_draft/config/secondary_experiments_v1.json"

cd "$REPO_ROOT"
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC

if [[ ! -d "$SWEEP_ROOT" ]]; then
  echo "Completed-training sweep root is missing: $SWEEP_ROOT" >&2
  exit 2
fi
if [[ ! -f "$SWEEP_ROOT/mode_marker_recovery_audit.json" ]]; then
  echo "Validated recovery audit is missing: $SWEEP_ROOT/mode_marker_recovery_audit.json" >&2
  exit 2
fi
if [[ -e "$SWEEP_ROOT/seed_sweep_status.csv" ]]; then
  echo "Merged sweep status already exists; refusing a second recovery." >&2
  exit 2
fi
for worker in $(seq 1 "$WORKERS"); do
  for path in \
    "$SWEEP_ROOT/worker_status_recovered_${worker}.csv" \
    "$SWEEP_ROOT/worker_logs/worker_${worker}_recovered.log"; do
    if [[ -e "$path" ]]; then
      echo "Recovery output already exists and will not be overwritten: $path" >&2
      exit 2
    fi
  done
done

"$TRAIN_PYTHON" - <<'PY'
import gymnasium
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
    raise SystemExit("Recovery requires the four-GPU training runtime.")
print("Recovery runtime:", torch.__version__, gymnasium.__version__,
      "GPUs", torch.cuda.device_count())
PY

declare -a pids=()
for worker in $(seq 1 "$WORKERS"); do
  gpu=$((worker - 1))
  status="$SWEEP_ROOT/worker_status_recovered_${worker}.csv"
  log="$SWEEP_ROOT/worker_logs/worker_${worker}_recovered.log"
  echo "Launching recovered sanity worker $worker/$WORKERS on CUDA device $gpu"
  env -u CONDA_PREFIX \
    CUDA_VISIBLE_DEVICES="$gpu" \
    RETICULATE_PYTHON="$TRAIN_PYTHON" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    VINE_SIM_CORES=1 \
    SWEEP_SEEDS_FILE="config/no_vine_ablation_seeds.yaml" \
    SWEEP_ROOT_DIR="$SWEEP_ROOT" \
    SWEEP_WORKER_COUNT="$WORKERS" \
    SWEEP_WORKER_INDEX="$worker" \
    SWEEP_STATUS_FILE="$status" \
    SWEEP_REUSE_COMPLETED_TRAINING=true \
    VINE_OBSERVATION_MODE=zero \
    "$RSCRIPT" --vanilla rl/run_seed_sweep.r config/config.yaml \
    >"$log" 2>&1 &
  pids+=("$!")
done

failed=0
for worker in $(seq 1 "$WORKERS"); do
  if ! wait "${pids[$((worker - 1))]}"; then
    echo "Recovered sanity worker $worker failed; see worker_${worker}_recovered.log" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one recovered sanity worker failed; evidence is preserved." >&2
  exit 1
fi

"$PYTHON" publication_pipeline_draft/secondary_experiment_protocol.py \
  merge-sweep-status \
  --contract "$CONTRACT" \
  --experiment no_vine_td3 \
  --inputs \
    "$SWEEP_ROOT/worker_status_recovered_1.csv" \
    "$SWEEP_ROOT/worker_status_recovered_2.csv" \
    "$SWEEP_ROOT/worker_status_recovered_3.csv" \
    "$SWEEP_ROOT/worker_status_recovered_4.csv" \
  --output "$SWEEP_ROOT/seed_sweep_status.csv"

echo "Recovered sanity gates and merged validation passed. Training was not rerun."
