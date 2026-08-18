#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gabirel/copula-portfolio-clean}"
PYTHON="${PYTHON:-/gabirel/miniforge3/bin/python3}"
RSCRIPT="${RSCRIPT:-/gabirel/miniforge3/bin/Rscript}"
POLICY_PYTHON="${POLICY_PYTHON:-/gabirel/venvs/copula-eval-torch271-cpu/bin/python}"
TERMINAL_WORKERS="${TERMINAL_WORKERS:-76}"

export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
# Bootstrap workers are processes.  Prevent each process from starting its own
# BLAS thread pool and oversubscribing the 80-core allocation.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$REPO_ROOT"

CONTRACT=publication_pipeline_draft/config/terminal_robustness_v1.json
RELEASE=frozen_releases/terminal_robustness_v1
RESULTS=analysis_outputs/terminal_robustness_v1
CLEANROOM=analysis_outputs/terminal_robustness_v1_cleanroom
VERIFICATION=analysis_outputs/terminal_robustness_v1_cleanroom_verification.json
ARCHIVE=terminal_robustness_v1_final.tar.gz
LOG=logs/terminal_robustness_v1.launch.log
PID_FILE=logs/terminal_robustness_v1.pid

require_file() {
  test -f "$1" || { echo "Missing required file: $1" >&2; exit 1; }
}

require_absent() {
  test ! -e "$1" || {
    echo "Refusing to overwrite immutable output: $1" >&2
    exit 1
  }
}

case "${1:-}" in
  validate)
    "$PYTHON" -m compileall -q publication_pipeline_draft
    "$PYTHON" -m pytest -q publication_pipeline_draft/tests
    "$RSCRIPT" --vanilla tests/run_tests.r
    "$RSCRIPT" --vanilla tests/test_publication_benchmarks.r
    POLICY_PYTHON="$POLICY_PYTHON" "$RSCRIPT" --vanilla \
      tests/test_policy_process_isolation.r
    "$PYTHON" -m publication_pipeline_draft.terminal_robustness_protocol \
      validate --contract "$CONTRACT"
    ;;
  inputs)
    "$PYTHON" - "$CONTRACT" <<'PY'
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
paths = [Path(contract["daily_adjusted_levels"])] + [
    Path(item["path"]) for item in contract["sources"]]
missing = [str(path) for path in paths if not path.is_file()]
for path in paths:
    print(("OK      " if path.is_file() else "MISSING ") + str(path))
if missing:
    raise SystemExit("Terminal campaign input failure; missing: " + ", ".join(missing))
PY
    ;;
  freeze)
    require_absent "$RELEASE"
    "$PYTHON" -m publication_pipeline_draft.terminal_robustness_protocol freeze \
      --repo-root . --contract "$CONTRACT" --output "$RELEASE"
    (cd "$RELEASE" && sha256sum -c CONTENTS.sha256)
    ;;
  run)
    require_file "$RELEASE/terminal_robustness_release_manifest.json"
    require_absent "$RESULTS"
    require_absent "$LOG"
    mkdir -p logs
    nohup "$PYTHON" -m publication_pipeline_draft.run_terminal_robustness \
      --release "$RELEASE" --output "$RESULTS" --workers "$TERMINAL_WORKERS" \
      > "$LOG" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Terminal robustness PID: $(cat "$PID_FILE")"
    echo "Follow with: tail -f $LOG"
    ;;
  status)
    if test -f "$PID_FILE"; then
      pid="$(cat "$PID_FILE")"
      ps -p "$pid" -o pid,etime,stat,cmd || true
    fi
    tail -n 40 "$LOG" || true
    if test -f "$RESULTS/terminal_robustness_manifest.json"; then
      "$PYTHON" -m json.tool "$RESULTS/terminal_robustness_manifest.json"
    fi
    ;;
  verify)
    "$PYTHON" -m publication_pipeline_draft.verify_terminal_robustness \
      --results "$RESULTS"
    (cd "$RESULTS" && sha256sum -c CONTENTS.sha256)
    ;;
  cleanroom)
    require_absent "$CLEANROOM"
    require_absent "$VERIFICATION"
    "$PYTHON" -m publication_pipeline_draft.run_terminal_robustness \
      --release "$RELEASE" --output "$CLEANROOM" --workers "$TERMINAL_WORKERS"
    "$PYTHON" -m publication_pipeline_draft.verify_terminal_robustness \
      --results "$RESULTS" --cleanroom-results "$CLEANROOM" \
      --output "$VERIFICATION"
    ;;
  finalize)
    require_file "$VERIFICATION"
    require_file "$RESULTS/terminal_robustness_manifest.json"
    require_absent "$ARCHIVE"
    require_absent "${ARCHIVE}.sha256"
    tar -czf "$ARCHIVE" "$RELEASE" "$RESULTS" "$VERIFICATION" \
      "$CONTRACT" \
      publication_pipeline_draft/terminal_robustness_protocol.py \
      publication_pipeline_draft/run_terminal_robustness.py \
      publication_pipeline_draft/verify_terminal_robustness.py \
      publication_pipeline_draft/tests/test_terminal_robustness_protocol.py \
      publication_pipeline_draft/TERMINAL_ROBUSTNESS_RUNBOOK.md \
      "$LOG"
    sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
    sha256sum -c "${ARCHIVE}.sha256"
    ;;
  *)
    echo "Usage: $0 {validate|inputs|freeze|run|status|verify|cleanroom|finalize}" >&2
    exit 2
    ;;
esac
