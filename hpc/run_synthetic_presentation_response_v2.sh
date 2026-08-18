#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_ROOT:-/gabirel/copula-portfolio-clean}"
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC

PYTHON="${PYTHON:-/gabirel/miniforge3/bin/python3}"
RSCRIPT="${RSCRIPT:-/gabirel/miniforge3/bin/Rscript}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/gabirel/miniforge3/envs/vine-rl/bin/python}"
POLICY_PYTHON="${POLICY_PYTHON:-/gabirel/venvs/copula-eval-torch271-cpu/bin/python}"
PRESENTATION_GPUS="${PRESENTATION_GPUS:-0,1,2,3}"
PRESENTATION_CPU_CORES="${PRESENTATION_CPU_CORES:-120}"

CONTRACT=publication_pipeline_draft/config/synthetic_presentation_response_v2.json
BUNDLE_ROOT=data/synthetic_presentation_response_v2
SOURCE_100=data/synthetic_dose_response_v1/vine_synthetic_100.RData
JOBS=protocol_manifests/synthetic_presentation_jobs_v2.csv
RELEASE=frozen_releases/synthetic_presentation_response_v2
RUN_ROOT=data/synthetic_presentation_response_runs_v2
STATUS=protocol_manifests/synthetic_presentation_sweep_status_v2.csv
LOG_ROOT=logs/synthetic_presentation_response_v2
AUDIT=analysis_outputs/synthetic_presentation_response_v2_audit
WEIGHTS=analysis_outputs/synthetic_presentation_response_v2_weights
RESULTS=analysis_outputs/synthetic_presentation_response_v2_results
DOSE100_WEIGHTS="${DOSE100_WEIGHTS:-analysis_outputs/synthetic_dose_response_v1_weights/synthetic_dose_policy_weight_manifest.csv}"
REALIZED="${PRESENTATION_REALIZED_PANEL:-locked_evaluation/main_oos_v4_operational_retry/inputs/realized_asset_gross.csv}"
CAUSAL_PANEL="${PRESENTATION_CAUSAL_PANEL:-analysis_outputs/post_hoc_compressed_vine_benchmark_reconciliation_v1/input_snapshots/causal_strategy_period_panel.csv}"
BENCHMARK_PANEL="${PRESENTATION_BENCHMARK_PANEL:-analysis_outputs/oos_v4_verified_770d2944/main_oos_v4_operational_retry/publication_results/raw/scored_monthly_panel.csv}"
ARCHIVE=synthetic_presentation_response_v2_final.tar.gz

case "${1:-}" in
  validate)
    "$PYTHON" -m compileall -q publication_pipeline_draft
    "$PYTHON" -m pytest -q publication_pipeline_draft/tests
    "$RSCRIPT" --vanilla tests/run_tests.r
    "$RSCRIPT" --vanilla tests/test_publication_benchmarks.r
    POLICY_PYTHON="$POLICY_PYTHON" "$RSCRIPT" --vanilla \
      tests/test_policy_process_isolation.r
    "$PYTHON" -m publication_pipeline_draft.synthetic_presentation_protocol \
      validate --repo-root . --contract "$CONTRACT"
    ;;
  bundle)
    test -f "$SOURCE_100"
    "$RSCRIPT" --vanilla rl/materialize_synthetic_presentation_bundle.r \
      "$SOURCE_100" \
      "$BUNDLE_ROOT/vine_synthetic_100_unique_1000_presentations.RData" 10
    (cd "$BUNDLE_ROOT" && sha256sum -c CONTENTS.sha256)
    ;;
  freeze)
    test -f protocol_manifests/training_python_runtime.json
    "$PYTHON" -m publication_pipeline_draft.synthetic_presentation_protocol \
      materialize-jobs --repo-root . --contract "$CONTRACT" \
      --output-root "$RUN_ROOT" --output "$JOBS"
    "$PYTHON" -m publication_pipeline_draft.synthetic_presentation_protocol freeze \
      --repo-root . --contract "$CONTRACT" --jobs "$JOBS" \
      --runtime protocol_manifests/training_python_runtime.json \
      --output "$RELEASE" --bundle "${RELEASE}.tar.gz"
    (cd frozen_releases && \
      sha256sum -c synthetic_presentation_response_v2.tar.gz.sha256)
    ;;
  train)
    mkdir -p logs
    nohup "$PYTHON" -m \
      publication_pipeline_draft.run_synthetic_presentation_sweep \
      --repo-root . --jobs "$JOBS" --release "$RELEASE" \
      --config config/config.yaml --train-python "$TRAIN_PYTHON" \
      --rscript "$RSCRIPT" --gpus "$PRESENTATION_GPUS" \
      --cpu-cores "$PRESENTATION_CPU_CORES" --log-root "$LOG_ROOT" \
      --status "$STATUS" \
      > logs/synthetic_presentation_response_v2.launch.log 2>&1 &
    echo $! > logs/synthetic_presentation_response_v2.pid
    echo "Presentation sweep PID: $(cat logs/synthetic_presentation_response_v2.pid)"
    ;;
  status)
    if test -f logs/synthetic_presentation_response_v2.pid; then
      pid="$(cat logs/synthetic_presentation_response_v2.pid)"
      ps -p "$pid" -o pid,etime,stat,cmd || true
    fi
    tail -n 30 logs/synthetic_presentation_response_v2.launch.log || true
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
      publication_pipeline_draft.audit_synthetic_presentation_sweep \
      --repo-root . --contract "$CONTRACT" --release "$RELEASE" \
      --jobs "$JOBS" --status "$STATUS" --output "$AUDIT"
    (cd "$AUDIT" && sha256sum -c CONTENTS.sha256)
    ;;
  replay)
    "$PYTHON" -m \
      publication_pipeline_draft.generate_synthetic_presentation_policy_weights \
      --repo-root . --contract "$CONTRACT" --release "$RELEASE" \
      --jobs "$JOBS" --audit "$AUDIT" --config config/config.yaml \
      --policy-python "$POLICY_PYTHON" --rscript "$RSCRIPT" \
      --workers 4 --output "$WEIGHTS"
    ;;
  analyze)
    test -f "$DOSE100_WEIGHTS"
    "$PYTHON" -m \
      publication_pipeline_draft.analyze_synthetic_presentation_response \
      --repo-root . --contract "$CONTRACT" \
      --weight-manifest \
        "$WEIGHTS/synthetic_presentation_policy_weight_manifest.csv" \
      --dose100-weight-manifest "$DOSE100_WEIGHTS" \
      --realized "$REALIZED" --causal-panel "$CAUSAL_PANEL" \
      --benchmark-panel "$BENCHMARK_PANEL" --output "$RESULTS"
    (cd "$RESULTS" && sha256sum -c CONTENTS.sha256)
    ;;
  finalize)
    test -f "$RESULTS/synthetic_presentation_analysis_manifest.json"
    test ! -e "$ARCHIVE"
    tar -czf "$ARCHIVE" \
      "$JOBS" "${JOBS}.sha256" "$STATUS" \
      "$RELEASE/synthetic_presentation_release_manifest.json" \
      "$AUDIT" "$WEIGHTS" "$RESULTS" \
      logs/synthetic_presentation_response_v2.launch.log
    sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
    sha256sum -c "${ARCHIVE}.sha256"
    ;;
  *)
    echo "Usage: $0 {validate|bundle|freeze|train|status|audit|replay|analyze|finalize}" >&2
    exit 2
    ;;
esac
