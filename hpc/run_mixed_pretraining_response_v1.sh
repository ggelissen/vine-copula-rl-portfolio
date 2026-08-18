#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_ROOT:-/gabirel/copula-portfolio-clean}"
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC

PYTHON="${PYTHON:-/gabirel/miniforge3/bin/python3}"
RSCRIPT="${RSCRIPT:-/gabirel/miniforge3/bin/Rscript}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/gabirel/miniforge3/envs/vine-rl/bin/python}"
POLICY_PYTHON="${POLICY_PYTHON:-/gabirel/venvs/copula-eval-torch271-cpu/bin/python}"
MIXED_GPUS="${MIXED_GPUS:-0,1,2,3}"
MIXED_CPU_CORES="${MIXED_CPU_CORES:-80}"

CONTRACT=publication_pipeline_draft/config/mixed_pretraining_response_v1.json
SOURCE=data/synthetic_dose_response_v1/vine_synthetic_100.RData
BUNDLE_ROOT=data/mixed_pretraining_response_v1
BUNDLE="$BUNDLE_ROOT/mixed_100synthetic_61historical_1000presentations.RData"
JOBS=protocol_manifests/mixed_pretraining_jobs_v1.csv
RELEASE=frozen_releases/mixed_pretraining_response_v1
RUN_ROOT=data/mixed_pretraining_runs_v1
STATUS=protocol_manifests/mixed_pretraining_sweep_status_v1.csv
LOG_ROOT=logs/mixed_pretraining_response_v1
AUDIT=analysis_outputs/mixed_pretraining_response_v1_audit
WEIGHTS=analysis_outputs/mixed_pretraining_response_v1_weights
RESULTS=analysis_outputs/mixed_pretraining_response_v1_results

SYNTHETIC_JOBS=protocol_manifests/synthetic_presentation_jobs_v2.csv
SYNTHETIC_AUDIT=analysis_outputs/synthetic_presentation_response_v2_audit
SYNTHETIC_WEIGHTS=analysis_outputs/synthetic_presentation_response_v2_weights/synthetic_presentation_policy_weight_manifest.csv
CONTROL_WEIGHTS=analysis_outputs/masked_pretraining_controls_v1_weights/masked_pretraining_control_weight_manifest.csv
REALIZED="${MIXED_REALIZED_PANEL:-locked_evaluation/main_oos_v4_operational_retry/inputs/realized_asset_gross.csv}"
ARCHIVE=mixed_pretraining_response_v1_final.tar.gz
CHECKPOINT_ARCHIVE=mixed_pretraining_response_v1_checkpoints.tar.gz
EVIDENCE_RELEASE=frozen_releases/mixed_pretraining_response_v1_evidence_v1
PUBLICATION_BUNDLE=manuscript_revision_causal_v1/publication_mixed_pretraining_v1

case "${1:-}" in
  validate)
    "$PYTHON" -m compileall -q publication_pipeline_draft
    "$PYTHON" -m pytest -q publication_pipeline_draft/tests
    "$RSCRIPT" --vanilla tests/run_tests.r
    "$RSCRIPT" --vanilla tests/test_publication_benchmarks.r
    POLICY_PYTHON="$POLICY_PYTHON" "$RSCRIPT" --vanilla \
      tests/test_policy_process_isolation.r
    "$PYTHON" -m publication_pipeline_draft.mixed_pretraining_protocol \
      validate --repo-root . --contract "$CONTRACT"
    ;;
  inputs)
    printf '%s  %s\n' \
      65eb5c715436f155c6cb8447d811e6cb96c2e9b55cc5b2d6ffeb560f9396b314 \
      "$SOURCE" | sha256sum -c -
    test -f "$SYNTHETIC_JOBS"
    test -f "$SYNTHETIC_AUDIT/synthetic_dose_audit_manifest.json"
    test -f "$SYNTHETIC_AUDIT/synthetic_dose_checkpoint_audit.csv"
    test -f "$SYNTHETIC_WEIGHTS"
    test -f "$CONTROL_WEIGHTS"
    test -f "$REALIZED"
    ;;
  bundle)
    test ! -e "$BUNDLE_ROOT"
    "$RSCRIPT" --vanilla rl/materialize_mixed_pretraining_bundle.r \
      "$SOURCE" "$BUNDLE" \
      65eb5c715436f155c6cb8447d811e6cb96c2e9b55cc5b2d6ffeb560f9396b314
    (cd "$BUNDLE_ROOT" && sha256sum -c CONTENTS.sha256)
    ;;
  freeze)
    test -f protocol_manifests/training_python_runtime.json
    "$PYTHON" -m publication_pipeline_draft.mixed_pretraining_protocol \
      materialize-jobs --repo-root . --contract "$CONTRACT" \
      --output-root "$RUN_ROOT" --output "$JOBS"
    "$PYTHON" -m publication_pipeline_draft.mixed_pretraining_protocol freeze \
      --repo-root . --contract "$CONTRACT" --jobs "$JOBS" \
      --runtime protocol_manifests/training_python_runtime.json \
      --output "$RELEASE" --bundle "${RELEASE}.tar.gz"
    (cd frozen_releases && sha256sum -c mixed_pretraining_response_v1.tar.gz.sha256)
    ;;
  train)
    mkdir -p logs
    nohup "$PYTHON" -m publication_pipeline_draft.run_mixed_pretraining_sweep \
      --repo-root . --jobs "$JOBS" --release "$RELEASE" \
      --config config/config.yaml --train-python "$TRAIN_PYTHON" \
      --rscript "$RSCRIPT" --gpus "$MIXED_GPUS" \
      --cpu-cores "$MIXED_CPU_CORES" --log-root "$LOG_ROOT" \
      --status "$STATUS" \
      > logs/mixed_pretraining_response_v1.launch.log 2>&1 &
    echo $! > logs/mixed_pretraining_response_v1.pid
    echo "Mixed sweep PID: $(cat logs/mixed_pretraining_response_v1.pid)"
    ;;
  status)
    if test -f logs/mixed_pretraining_response_v1.pid; then
      pid="$(cat logs/mixed_pretraining_response_v1.pid)"
      ps -p "$pid" -o pid,etime,stat,cmd || true
    fi
    tail -n 30 logs/mixed_pretraining_response_v1.launch.log || true
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
      publication_pipeline_draft.audit_mixed_pretraining_comparison \
      --repo-root . --contract "$CONTRACT" --release "$RELEASE" \
      --jobs "$JOBS" --status "$STATUS" \
      --synthetic-audit "$SYNTHETIC_AUDIT" \
      --synthetic-jobs "$SYNTHETIC_JOBS" --output "$AUDIT"
    (cd "$AUDIT" && sha256sum -c CONTENTS.sha256)
    ;;
  replay)
    "$PYTHON" -m \
      publication_pipeline_draft.generate_mixed_pretraining_comparison_weights \
      --repo-root . --contract "$CONTRACT" --release "$RELEASE" \
      --jobs "$JOBS" --audit "$AUDIT" --config config/config.yaml \
      --policy-python "$POLICY_PYTHON" --rscript "$RSCRIPT" \
      --workers 4 --output "$WEIGHTS"
    ;;
  analyze)
    "$PYTHON" -m publication_pipeline_draft.analyze_mixed_pretraining_response \
      --repo-root . --contract "$CONTRACT" \
      --comparison-weight-manifest \
        "$WEIGHTS/mixed_pretraining_comparison_weight_manifest.csv" \
      --synthetic-weight-manifest "$SYNTHETIC_WEIGHTS" \
      --control-weight-manifest "$CONTROL_WEIGHTS" \
      --realized "$REALIZED" --output "$RESULTS"
    (cd "$RESULTS" && sha256sum -c CONTENTS.sha256)
    ;;
  checkpoint-archive)
    test -f "$AUDIT/mixed_pretraining_audit_manifest.json"
    test ! -e "$CHECKPOINT_ARCHIVE"
    tar -czf "$CHECKPOINT_ARCHIVE" "$RUN_ROOT" "$AUDIT"
    sha256sum "$CHECKPOINT_ARCHIVE" > "${CHECKPOINT_ARCHIVE}.sha256"
    sha256sum -c "${CHECKPOINT_ARCHIVE}.sha256"
    ;;
  finalize)
    test -f "$RESULTS/mixed_pretraining_analysis_manifest.json"
    test ! -e "$ARCHIVE"
    tar -czf "$ARCHIVE" \
      "$JOBS" "${JOBS}.sha256" "$STATUS" \
      "$RELEASE/mixed_pretraining_release_manifest.json" \
      "$AUDIT" "$WEIGHTS" "$RESULTS" "$CONTRACT" \
      logs/mixed_pretraining_response_v1.launch.log
    sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
    sha256sum -c "${ARCHIVE}.sha256"
    ;;
  freeze-evidence)
    test ! -e "$EVIDENCE_RELEASE"
    "$PYTHON" -m publication_pipeline_draft.freeze_mixed_pretraining_evidence \
      --final-archive "$ARCHIVE" \
      --final-checksum "${ARCHIVE}.sha256" \
      --checkpoint-archive "$CHECKPOINT_ARCHIVE" \
      --checkpoint-checksum "${CHECKPOINT_ARCHIVE}.sha256" \
      --output "$EVIDENCE_RELEASE"
    (cd "$EVIDENCE_RELEASE" && sha256sum -c CONTENTS.sha256)
    ;;
  publication)
    test -f "$EVIDENCE_RELEASE/mixed_pretraining_evidence_manifest.json"
    "$PYTHON" -m \
      publication_pipeline_draft.generate_mixed_pretraining_publication_artifacts \
      --repo-root . --evidence-release "$EVIDENCE_RELEASE" \
      --realized "$REALIZED" --output "$PUBLICATION_BUNDLE"
    (cd "$PUBLICATION_BUNDLE" && sha256sum -c CONTENTS.sha256)
    ;;
  *)
    echo "Usage: $0 {validate|inputs|bundle|freeze|train|status|audit|replay|analyze|checkpoint-archive|finalize|freeze-evidence|publication}" >&2
    exit 2
    ;;
esac
