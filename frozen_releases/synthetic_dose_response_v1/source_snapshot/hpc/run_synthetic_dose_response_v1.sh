#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_ROOT:-/gabirel/copula-portfolio-clean}"
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC

PYTHON="${PYTHON:-/gabirel/miniforge3/bin/python3}"
RSCRIPT="${RSCRIPT:-/gabirel/miniforge3/bin/Rscript}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/gabirel/miniforge3/envs/vine-rl/bin/python}"
POLICY_PYTHON="${POLICY_PYTHON:-/gabirel/venvs/copula-eval-torch271-cpu/bin/python}"
DOSE_GPUS="${DOSE_GPUS:-0,1,2,3}"
DOSE_CPU_CORES="${DOSE_CPU_CORES:-120}"

CONTRACT=publication_pipeline_draft/config/synthetic_dose_response_v1.json
BUNDLE_ROOT=data/synthetic_dose_response_v1
JOBS=protocol_manifests/synthetic_dose_jobs_v1.csv
RELEASE=frozen_releases/synthetic_dose_response_v1
RUN_ROOT=data/synthetic_dose_response_runs_v1
STATUS=protocol_manifests/synthetic_dose_sweep_status_v1.csv
LOG_ROOT=logs/synthetic_dose_response_v1
AUDIT=analysis_outputs/synthetic_dose_response_v1_audit
WEIGHTS=analysis_outputs/synthetic_dose_response_v1_weights
RESULTS=analysis_outputs/synthetic_dose_response_v1_results
REALIZED="${DOSE_REALIZED_PANEL:-locked_evaluation/main_oos_v4_operational_retry/inputs/realized_asset_gross.csv}"
CAUSAL_PANEL="${DOSE_CAUSAL_PANEL:-analysis_outputs/post_hoc_compressed_vine_benchmark_reconciliation_v1/input_snapshots/causal_strategy_period_panel.csv}"
BENCHMARK_PANEL="${DOSE_BENCHMARK_PANEL:-analysis_outputs/oos_v4_verified_770d2944/main_oos_v4_operational_retry/publication_results/raw/scored_monthly_panel.csv}"

case "${1:-}" in
  validate)
    "$PYTHON" -m compileall -q publication_pipeline_draft
    "$PYTHON" -m pytest -q publication_pipeline_draft/tests
    "$RSCRIPT" --vanilla tests/run_tests.r
    "$RSCRIPT" --vanilla tests/test_publication_benchmarks.r
    POLICY_PYTHON="$POLICY_PYTHON" "$RSCRIPT" --vanilla \
      tests/test_policy_process_isolation.r
    "$PYTHON" -m publication_pipeline_draft.synthetic_dose_protocol validate \
      --repo-root . --contract "$CONTRACT"
    ;;
  bundle)
    "$RSCRIPT" --vanilla rl/materialize_synthetic_dose_bundle.r \
      data/synthetic_returns.RData \
      "$BUNDLE_ROOT/vine_synthetic_100.RData" 100
    (cd "$BUNDLE_ROOT" && sha256sum -c CONTENTS.sha256)
    ;;
  freeze)
    "$PYTHON" -m publication_pipeline_draft.synthetic_dose_protocol \
      materialize-jobs --repo-root . --contract "$CONTRACT" \
      --output-root "$RUN_ROOT" --output "$JOBS"
    "$PYTHON" -m publication_pipeline_draft.synthetic_dose_protocol freeze \
      --repo-root . --contract "$CONTRACT" --jobs "$JOBS" \
      --runtime protocol_manifests/training_python_runtime.json \
      --output "$RELEASE" --bundle "${RELEASE}.tar.gz"
    (cd frozen_releases && sha256sum -c synthetic_dose_response_v1.tar.gz.sha256)
    ;;
  train)
    nohup "$PYTHON" -m publication_pipeline_draft.run_synthetic_dose_sweep \
      --repo-root . --jobs "$JOBS" --release "$RELEASE" \
      --config config/config.yaml --train-python "$TRAIN_PYTHON" \
      --rscript "$RSCRIPT" --gpus "$DOSE_GPUS" --cpu-cores "$DOSE_CPU_CORES" \
      --log-root "$LOG_ROOT" --status "$STATUS" \
      > logs/synthetic_dose_response_v1.launch.log 2>&1 &
    echo $! > logs/synthetic_dose_response_v1.pid
    echo "Synthetic-dose sweep PID: $(cat logs/synthetic_dose_response_v1.pid)"
    ;;
  audit)
    "$TRAIN_PYTHON" -m publication_pipeline_draft.audit_synthetic_dose_sweep \
      --repo-root . --contract "$CONTRACT" --release "$RELEASE" \
      --jobs "$JOBS" --status "$STATUS" --output "$AUDIT"
    (cd "$AUDIT" && sha256sum -c CONTENTS.sha256)
    ;;
  replay)
    "$PYTHON" -m publication_pipeline_draft.generate_synthetic_dose_policy_weights \
      --repo-root . --contract "$CONTRACT" --release "$RELEASE" \
      --jobs "$JOBS" --audit "$AUDIT" --config config/config.yaml \
      --policy-python "$POLICY_PYTHON" --rscript "$RSCRIPT" \
      --workers 4 --output "$WEIGHTS"
    ;;
  analyze)
    "$PYTHON" -m publication_pipeline_draft.analyze_synthetic_dose_response \
      --repo-root . --contract "$CONTRACT" \
      --weight-manifest "$WEIGHTS/synthetic_dose_policy_weight_manifest.csv" \
      --realized "$REALIZED" --causal-panel "$CAUSAL_PANEL" \
      --benchmark-panel "$BENCHMARK_PANEL" --output "$RESULTS"
    (cd "$RESULTS" && sha256sum -c CONTENTS.sha256)
    ;;
  *)
    echo "Usage: $0 {validate|bundle|freeze|train|audit|replay|analyze}" >&2
    exit 2
    ;;
esac
