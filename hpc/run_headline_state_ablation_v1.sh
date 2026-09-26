#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_ROOT:-/gabirel/copula-portfolio-clean}"
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC

PYTHON="${PYTHON:-/gabirel/miniforge3/bin/python3}"
RSCRIPT="${RSCRIPT:-/gabirel/miniforge3/bin/Rscript}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/gabirel/miniforge3/envs/vine-rl/bin/python}"
POLICY_PYTHON="${POLICY_PYTHON:-/gabirel/venvs/copula-eval-torch271-cpu/bin/python}"
GPUS="${HEADLINE_STATE_GPUS:-0,1,2,3}"
CPU_CORES="${HEADLINE_STATE_CPU_CORES:-80}"

CONTRACT=publication_pipeline_draft/config/headline_state_ablation_v1.json
JOBS=protocol_manifests/headline_state_jobs_v1.csv
RELEASE=frozen_releases/headline_state_ablation_v1
STATUS=protocol_manifests/headline_state_status_v1.csv
LOG_ROOT=logs/headline_state_ablation_v1_workers
LAUNCH_LOG=logs/headline_state_ablation_v1.launch.log
AUDIT=analysis_outputs/headline_state_ablation_v1_audit
WEIGHTS=analysis_outputs/headline_state_ablation_v1_weights
RESULTS=analysis_outputs/headline_state_ablation_v1_results
REALIZED="${HEADLINE_STATE_REALIZED:-locked_evaluation/main_oos_v4_operational_retry/inputs/realized_asset_gross.csv}"
FINAL=headline_state_ablation_v1_final.tar.gz
CHECKPOINTS=headline_state_ablation_v1_checkpoints.tar.gz

case "${1:-}" in
  validate)
    "$PYTHON" -m pytest -q publication_pipeline_draft/tests
    "$RSCRIPT" --vanilla tests/run_tests.r
    "$RSCRIPT" --vanilla tests/test_publication_benchmarks.r
    POLICY_PYTHON="$POLICY_PYTHON" "$RSCRIPT" --vanilla \
      tests/test_policy_process_isolation.r
    "$PYTHON" -m publication_pipeline_draft.headline_state_ablation validate
    ;;
  prepare)
    test -f protocol_manifests/training_python_runtime.json
    test -f "$REALIZED"
    "$PYTHON" -m publication_pipeline_draft.headline_state_ablation prepare \
      --jobs "$JOBS" --output "$RELEASE" \
      --archive "${RELEASE}.tar.gz" \
      --runtime protocol_manifests/training_python_runtime.json
    (cd "$RELEASE" && sha256sum -c CONTENTS.sha256)
    ;;
  train)
    test -f "$JOBS"
    test -d "$RELEASE"
    test ! -e "$STATUS"
    test ! -e "$LOG_ROOT"
    mkdir -p logs
    nohup "$PYTHON" -m publication_pipeline_draft.headline_state_ablation train \
      --jobs "$JOBS" --release "$RELEASE" \
      --train-python "$TRAIN_PYTHON" --rscript "$RSCRIPT" \
      --gpus "$GPUS" --cpu-cores "$CPU_CORES" \
      --log-root "$LOG_ROOT" --status "$STATUS" \
      > "$LAUNCH_LOG" 2>&1 &
    echo $! > logs/headline_state_ablation_v1.pid
    echo "Training PID: $(cat logs/headline_state_ablation_v1.pid)"
    ;;
  status)
    if test -f logs/headline_state_ablation_v1.pid; then
      ps -p "$(cat logs/headline_state_ablation_v1.pid)" -o pid,etime,stat,cmd || true
    fi
    tail -n 35 "$LAUNCH_LOG" || true
    if test -f "$STATUS"; then
      "$PYTHON" -c 'import csv,sys; x=list(csv.DictReader(open(sys.argv[1]))); print("completed:",len(x),"passed:",sum(r["passed"].lower()=="true" for r in x))' "$STATUS"
    else
      echo "Final status has not been written yet."
    fi
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader || true
    ;;
  audit)
    "$TRAIN_PYTHON" -m publication_pipeline_draft.headline_state_ablation audit \
      --jobs "$JOBS" --release "$RELEASE" --status "$STATUS" \
      --output "$AUDIT"
    (cd "$AUDIT" && sha256sum -c CONTENTS.sha256)
    ;;
  replay)
    "$PYTHON" -m publication_pipeline_draft.headline_state_ablation replay \
      --jobs "$JOBS" --release "$RELEASE" --audit "$AUDIT" \
      --policy-python "$POLICY_PYTHON" --rscript "$RSCRIPT" \
      --workers 4 --output "$WEIGHTS"
    ;;
  analyze)
    "$PYTHON" -m publication_pipeline_draft.analyze_headline_state_ablation \
      --jobs "$JOBS" --release "$RELEASE" --audit "$AUDIT" \
      --weights "$WEIGHTS" --realized "$REALIZED" --output "$RESULTS"
    (cd "$RESULTS" && sha256sum -c CONTENTS.sha256)
    ;;
  checkpoint-archive)
    test -f "$AUDIT/headline_state_audit_manifest.json"
    test ! -e "$CHECKPOINTS"
    tar -czf "$CHECKPOINTS" data/headline_state_ablation_runs_v1 "$AUDIT"
    sha256sum "$CHECKPOINTS" > "${CHECKPOINTS}.sha256"
    sha256sum -c "${CHECKPOINTS}.sha256"
    ;;
  finalize)
    test -f "$RESULTS/headline_state_analysis_manifest.json"
    test ! -e "$FINAL"
    tar -czf "$FINAL" "$CONTRACT" "$JOBS" "$STATUS" \
      "$RELEASE" frozen_releases/mixed_pretraining_response_v1 \
      frozen_releases/mixed_pretraining_response_v1_evidence_v1 \
      data/mixed_pretraining_response_v1/mixed_pretraining_bundle_manifest.csv \
      "$REALIZED" "$AUDIT" "$WEIGHTS" "$RESULTS" "$LAUNCH_LOG"
    sha256sum "$FINAL" > "${FINAL}.sha256"
    sha256sum -c "${FINAL}.sha256"
    ;;
  *)
    echo "Usage: $0 {validate|prepare|train|status|audit|replay|analyze|checkpoint-archive|finalize}" >&2
    exit 2
    ;;
esac
