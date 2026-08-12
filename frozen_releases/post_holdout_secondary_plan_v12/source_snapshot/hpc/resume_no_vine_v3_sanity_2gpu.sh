#!/usr/bin/env bash
# Resume only no-holdout sanity for the completed clean v3 checkpoints.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd -P)}"
RSCRIPT="${RSCRIPT:-Rscript}"
PYTHON="${PYTHON:-python3}"
: "${TRAIN_PYTHON:?Set TRAIN_PYTHON to the exact GPU training interpreter}"
SWEEP_ROOT="${NO_VINE_SWEEP_ROOT:-$REPO_ROOT/data/no_vine_rl_runs_secondary_v3}"
WORKERS="${NO_VINE_WORKERS:-2}"
CONTRACT="$REPO_ROOT/publication_pipeline_draft/config/secondary_experiments_v1.json"
SEEDS="20260841,20260842,20260843,20260844,20260845,20260846,20260847,20260848,20260849,20260850"

cd "$REPO_ROOT"
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC

if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NO_VINE_WORKERS must be a positive integer." >&2
  exit 2
fi
if [[ ! -d "$SWEEP_ROOT" ]]; then
  echo "Completed v3 sweep root is missing: $SWEEP_ROOT" >&2
  exit 2
fi
if [[ -e "$SWEEP_ROOT/seed_sweep_status.csv" ]]; then
  echo "Merged status already exists; refusing to overwrite it." >&2
  exit 2
fi
if find "$SWEEP_ROOT" -mindepth 2 -maxdepth 2 -type d \
    -name sanity_no_holdout -print -quit | grep -q .; then
  echo "A sanity_no_holdout directory already exists; inspect it before retrying." >&2
  exit 2
fi
if [[ -e "$SWEEP_ROOT/checkpoint_evidence.json" ]]; then
  echo "Checkpoint evidence already exists; refusing an ambiguous second resume." >&2
  exit 2
fi
for worker in $(seq 1 "$WORKERS"); do
  for path in \
    "$SWEEP_ROOT/worker_status_sanity_retry1_${worker}.csv" \
    "$SWEEP_ROOT/worker_logs/worker_${worker}_sanity_retry1.log"; do
    if [[ -e "$path" ]]; then
      echo "Sanity-retry output already exists: $path" >&2
      exit 2
    fi
  done
done

NO_VINE_WORKERS="$WORKERS" "$TRAIN_PYTHON" - <<'PY'
import os
import gymnasium
import pandas
import torch

required_gpus = int(os.environ["NO_VINE_WORKERS"])
if not torch.cuda.is_available() or torch.cuda.device_count() < required_gpus:
    raise SystemExit(
        f"Sanity resume requires {required_gpus} visible GPUs; "
        f"PyTorch sees {torch.cuda.device_count()}."
    )
print(
    "Sanity runtime:", torch.__version__, gymnasium.__version__,
    pandas.__version__, "GPUs", torch.cuda.device_count()
)
PY

"$PYTHON" publication_pipeline_draft/preflight_no_vine_training_contract.py \
  --repo-root "$REPO_ROOT" --sweep-root "$SWEEP_ROOT" >/dev/null

attestation_temp="$(mktemp /tmp/no-vine-v3-checkpoints.XXXXXX.json)"
trap 'rm -f "$attestation_temp"' EXIT
"$TRAIN_PYTHON" publication_pipeline_draft/verify_no_vine_training_evidence.py \
  --sweep-root "$SWEEP_ROOT" --seeds "$SEEDS" --require-embedded \
  >"$attestation_temp"
mv "$attestation_temp" "$SWEEP_ROOT/checkpoint_evidence.json"

declare -a pids=()
for worker in $(seq 1 "$WORKERS"); do
  gpu=$((worker - 1))
  status="$SWEEP_ROOT/worker_status_sanity_retry1_${worker}.csv"
  log="$SWEEP_ROOT/worker_logs/worker_${worker}_sanity_retry1.log"
  echo "Launching sanity-only worker $worker/$WORKERS on CUDA device $gpu"
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
    echo "Sanity worker $worker failed; see worker_${worker}_sanity_retry1.log" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one sanity-only worker failed; all evidence is preserved." >&2
  exit 1
fi

status_inputs=()
for worker in $(seq 1 "$WORKERS"); do
  status_inputs+=("$SWEEP_ROOT/worker_status_sanity_retry1_${worker}.csv")
done
"$PYTHON" publication_pipeline_draft/secondary_experiment_protocol.py \
  merge-sweep-status --contract "$CONTRACT" --experiment no_vine_td3 \
  --inputs "${status_inputs[@]}" \
  --output "$SWEEP_ROOT/seed_sweep_status.csv"

echo "Sanity-only resume passed. Training was not rerun."
