#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd -P)}"
RSCRIPT="${RSCRIPT:-Rscript}"
PYTHON="${PYTHON:-python3}"
: "${TRAIN_PYTHON:?Set TRAIN_PYTHON to the GPU-enabled reticulate Python for training}"
export TRAIN_PYTHON
V4_ARCHIVE="${V4_ARCHIVE:-locked_evaluation/main_oos_v4_operational_retry.tar.gz}"
V4_SIDECAR="${V4_SIDECAR:-locked_evaluation/main_oos_v4_operational_retry.tar.gz.sha256}"
PLAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$REPO_ROOT"
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
sha256sum -c "$PLAN_ROOT/LIVE_SOURCE_CONTENTS.sha256"

# These runs are post-holdout explanatory. They cannot retroactively
# convert locked_oos_v1 into a fresh confirmatory result.

# 1. Matched-capacity TD3 without policy-visible vine state (expensive).
# Four independent one-GPU workers; defaults to 18 vine-simulation cores each.
NO_VINE_SWEEP_ROOT='data/no_vine_rl_runs_secondary_v2' RSCRIPT="$RSCRIPT" PYTHON="$PYTHON" \
  bash hpc/run_no_vine_4gpu.sh

# 2. Fail-closed validation before aggregation.
"$PYTHON" publication_pipeline_draft/secondary_experiment_protocol.py \
  validate-sweep \
  --contract publication_pipeline_draft/config/secondary_experiments_v1.json \
  --experiment no_vine_td3 \
  --status 'data/no_vine_rl_runs_secondary_v2/seed_sweep_status.csv'

# 3. Aggregate/freeze only after all expected gates pass.
"$PYTHON" publication_pipeline_draft/diagnostic_artifacts.py \
  --rl-runs 'data/no_vine_rl_runs_secondary_v2' --expected-seeds 10 \
  --output data/publication_no_vine_training_artifacts_10seeds
tar -czf no_vine_training_artifacts_10seeds.tar.gz \
  -C data publication_no_vine_training_artifacts_10seeds
"$PYTHON" publication_pipeline_draft/freeze_training_release.py \
  --repo-root . --rl-runs 'data/no_vine_rl_runs_secondary_v2' \
  --diagnostics-archive "$REPO_ROOT/no_vine_training_artifacts_10seeds.tar.gz" \
  --expected-seeds 10 \
  --output frozen_releases/no_vine_schema5_secondary_v1 \
  --bundle frozen_releases/no_vine_schema5_secondary_v1.tar.gz
(cd frozen_releases && sha256sum -c no_vine_schema5_secondary_v1.tar.gz.sha256)

# 4. Validate that every frozen full seed contains the already-trained
# pre-fine-tuning checkpoint. No training or holdout scoring occurs here.
"$PYTHON" publication_pipeline_draft/secondary_experiment_protocol.py \
  validate-checkpoints \
  --contract publication_pipeline_draft/config/secondary_experiments_v1.json \
  --experiment pretrained_only \
  --training-release frozen_releases/training_schema5_v1

# 5. Same-sample explanatory evaluation. This is never confirmatory.
: "${POLICY_PYTHON:?Set POLICY_PYTHON to the isolated CPU PyTorch interpreter}"
export POLICY_PYTHON
"$PYTHON" publication_pipeline_draft/secondary_ablation_batch.py \
  --repo-root . \
  --contract publication_pipeline_draft/config/secondary_evaluation_contract_v1.json \
  --successful-archive "$V4_ARCHIVE" \
  --successful-sidecar "$V4_SIDECAR" \
  --evaluation-contract publication_pipeline_draft/config/evaluation_contract.json \
  --runtime-config config/config.yaml \
  --full-training-release frozen_releases/training_schema5_v1 \
  --no-vine-training-release frozen_releases/no_vine_schema5_secondary_v1 \
  --secondary-plan-release "$PLAN_ROOT" \
  --output secondary_evaluation/post_holdout_explanatory_ablation_v1 \
  --bundle secondary_evaluation/post_holdout_explanatory_ablation_v1.tar.gz \
  --rscript "$RSCRIPT"
(cd secondary_evaluation && \
  sha256sum -c post_holdout_explanatory_ablation_v1.tar.gz.sha256)

echo 'Secondary explanatory batch complete. Do not present it as a fresh confirmatory result.'
