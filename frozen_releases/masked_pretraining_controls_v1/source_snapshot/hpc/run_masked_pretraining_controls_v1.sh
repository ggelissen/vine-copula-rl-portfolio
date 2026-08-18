#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_ROOT:-/gabirel/copula-portfolio-clean}"
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC

PYTHON="${PYTHON:-/gabirel/miniforge3/bin/python3}"
RSCRIPT="${RSCRIPT:-/gabirel/miniforge3/bin/Rscript}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/gabirel/miniforge3/envs/vine-rl/bin/python}"
POLICY_PYTHON="${POLICY_PYTHON:-/gabirel/venvs/copula-eval-torch271-cpu/bin/python}"
CONTROL_GPUS="${CONTROL_GPUS:-0,1,2,3}"
CONTROL_CPU_CORES="${CONTROL_CPU_CORES:-120}"

CONTRACT=publication_pipeline_draft/config/masked_pretraining_controls_v1.json
JOBS=protocol_manifests/masked_pretraining_control_jobs_v1.csv
RELEASE=frozen_releases/masked_pretraining_controls_v1
RUN_ROOT=data/masked_pretraining_control_runs_v1
STATUS=protocol_manifests/masked_pretraining_control_sweep_status_v1.csv
LOG_ROOT=logs/masked_pretraining_controls_v1
AUDIT=analysis_outputs/masked_pretraining_controls_v1_audit
WEIGHTS=analysis_outputs/masked_pretraining_controls_v1_weights
RESULTS=analysis_outputs/masked_pretraining_controls_v1_results
CANDIDATE_WEIGHTS="${MASKED_CANDIDATE_WEIGHTS:-analysis_outputs/synthetic_presentation_response_v2_weights/synthetic_presentation_policy_weight_manifest.csv}"
REALIZED="${MASKED_CONTROL_REALIZED_PANEL:-locked_evaluation/main_oos_v4_operational_retry/inputs/realized_asset_gross.csv}"
CAUSAL_PANEL="${MASKED_CONTROL_CAUSAL_PANEL:-analysis_outputs/post_hoc_compressed_vine_benchmark_reconciliation_v1/input_snapshots/causal_strategy_period_panel.csv}"
BENCHMARK_PANEL="${MASKED_CONTROL_BENCHMARK_PANEL:-analysis_outputs/oos_v4_verified_770d2944/main_oos_v4_operational_retry/publication_results/raw/scored_monthly_panel.csv}"
ARCHIVE=masked_pretraining_controls_v1_final.tar.gz
CHECKPOINT_ARCHIVE=masked_pretraining_controls_v1_checkpoints.tar.gz

case "${1:-}" in
  validate)
    "$PYTHON" -m compileall -q publication_pipeline_draft
    "$PYTHON" -m pytest -q publication_pipeline_draft/tests
    "$RSCRIPT" --vanilla tests/run_tests.r
    "$RSCRIPT" --vanilla tests/test_publication_benchmarks.r
    POLICY_PYTHON="$POLICY_PYTHON" "$RSCRIPT" --vanilla \
      tests/test_policy_process_isolation.r
    "$PYTHON" -m \
      publication_pipeline_draft.masked_pretraining_controls_protocol \
      validate --repo-root . --contract "$CONTRACT"
    ;;
  inputs)
    test -f data/ablation_training_bundles/historical_prefix_repeated.RData
    test -f data/ablation_training_bundles/moving_block_bootstrap.RData
    printf '%s  %s\n' \
      1f07f655064ccb33dac3c60b2d1ca16ad2c91c73509313de46caeaa15b99e52e \
      data/ablation_training_bundles/historical_prefix_repeated.RData \
      0f82cd46391b4c7e08e34470826fef3886cb14a8f3cbf67966a70af36a6bf2a9 \
      data/ablation_training_bundles/moving_block_bootstrap.RData | sha256sum -c -
    test -f "$CANDIDATE_WEIGHTS"
    printf '%s  %s\n' \
      96a93e6f7b48a23f40def86a7f23c99c2129b411272198962d859fe8628783fe \
      "$CANDIDATE_WEIGHTS" | sha256sum -c -
    ;;
  freeze)
    test -f protocol_manifests/training_python_runtime.json
    "$PYTHON" -m \
      publication_pipeline_draft.masked_pretraining_controls_protocol \
      materialize-jobs --repo-root . --contract "$CONTRACT" \
      --output-root "$RUN_ROOT" --output "$JOBS"
    "$PYTHON" -m \
      publication_pipeline_draft.masked_pretraining_controls_protocol freeze \
      --repo-root . --contract "$CONTRACT" --jobs "$JOBS" \
      --runtime protocol_manifests/training_python_runtime.json \
      --output "$RELEASE" --bundle "${RELEASE}.tar.gz"
    (cd frozen_releases && \
      sha256sum -c masked_pretraining_controls_v1.tar.gz.sha256)
    ;;
  train)
    mkdir -p logs
    nohup "$PYTHON" -m \
      publication_pipeline_draft.run_masked_pretraining_controls \
      --repo-root . --jobs "$JOBS" --release "$RELEASE" \
      --config config/config.yaml --train-python "$TRAIN_PYTHON" \
      --rscript "$RSCRIPT" --gpus "$CONTROL_GPUS" \
      --cpu-cores "$CONTROL_CPU_CORES" --log-root "$LOG_ROOT" \
      --status "$STATUS" \
      > logs/masked_pretraining_controls_v1.launch.log 2>&1 &
    echo $! > logs/masked_pretraining_controls_v1.pid
    echo "Masked-control sweep PID: $(cat logs/masked_pretraining_controls_v1.pid)"
    ;;
  status)
    if test -f logs/masked_pretraining_controls_v1.pid; then
      pid="$(cat logs/masked_pretraining_controls_v1.pid)"
      ps -p "$pid" -o pid,etime,stat,cmd || true
    fi
    tail -n 30 logs/masked_pretraining_controls_v1.launch.log || true
    if test -f "$STATUS"; then
      "$PYTHON" -c \
        'import pandas as p,sys; x=p.read_csv(sys.argv[1]); print(x.groupby(["experiment_id","passed"]).size())' \
        "$STATUS"
    else
      echo "Status file not written yet; workers are still running."
    fi
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
      --format=csv,noheader || true
    ;;
  audit)
    "$TRAIN_PYTHON" -m \
      publication_pipeline_draft.audit_masked_pretraining_controls \
      --repo-root . --contract "$CONTRACT" --release "$RELEASE" \
      --jobs "$JOBS" --status "$STATUS" --output "$AUDIT"
    (cd "$AUDIT" && sha256sum -c CONTENTS.sha256)
    ;;
  replay)
    "$PYTHON" -m \
      publication_pipeline_draft.generate_masked_pretraining_control_weights \
      --repo-root . --contract "$CONTRACT" --release "$RELEASE" \
      --jobs "$JOBS" --audit "$AUDIT" --config config/config.yaml \
      --policy-python "$POLICY_PYTHON" --rscript "$RSCRIPT" \
      --workers 4 --output "$WEIGHTS"
    ;;
  analyze)
    test -f "$CANDIDATE_WEIGHTS"
    "$PYTHON" -m \
      publication_pipeline_draft.analyze_masked_pretraining_controls \
      --repo-root . --contract "$CONTRACT" \
      --weight-manifest \
        "$WEIGHTS/masked_pretraining_control_weight_manifest.csv" \
      --candidate-weight-manifest "$CANDIDATE_WEIGHTS" \
      --realized "$REALIZED" --causal-panel "$CAUSAL_PANEL" \
      --benchmark-panel "$BENCHMARK_PANEL" --output "$RESULTS"
    (cd "$RESULTS" && sha256sum -c CONTENTS.sha256)
    ;;
  checkpoint-archive)
    test -f "$AUDIT/synthetic_dose_audit_manifest.json"
    test ! -e "$CHECKPOINT_ARCHIVE"
    tar -czf "$CHECKPOINT_ARCHIVE" "$RUN_ROOT" "$AUDIT"
    sha256sum "$CHECKPOINT_ARCHIVE" > "${CHECKPOINT_ARCHIVE}.sha256"
    sha256sum -c "${CHECKPOINT_ARCHIVE}.sha256"
    ;;
  finalize)
    test -f "$RESULTS/masked_pretraining_analysis_manifest.json"
    test ! -e "$ARCHIVE"
    tar -czf "$ARCHIVE" \
      "$JOBS" "${JOBS}.sha256" "$STATUS" \
      "$RELEASE/masked_pretraining_control_release_manifest.json" \
      "$AUDIT" "$WEIGHTS" "$RESULTS" \
      logs/masked_pretraining_controls_v1.launch.log
    sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
    sha256sum -c "${ARCHIVE}.sha256"
    ;;
  *)
    echo "Usage: $0 {validate|inputs|freeze|train|status|audit|replay|analyze|checkpoint-archive|finalize}" >&2
    exit 2
    ;;
esac
