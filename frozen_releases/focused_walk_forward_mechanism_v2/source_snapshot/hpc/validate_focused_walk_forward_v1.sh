#!/usr/bin/env bash
# Fail-closed validation for the prospective focused walk-forward framework.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd -P)}"
PYTHON="${PYTHON:-/gabirel/miniforge3/bin/python3}"
RSCRIPT="${RSCRIPT:-/gabirel/miniforge3/bin/Rscript}"
: "${POLICY_PYTHON:?Set POLICY_PYTHON to the isolated policy-inference interpreter}"

cd "$REPO_ROOT"
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC

"$PYTHON" -m compileall -q publication_pipeline_draft
"$PYTHON" -m pytest -q \
  publication_pipeline_draft/tests/test_focused_walk_forward_protocol.py
"$PYTHON" -m pytest -q publication_pipeline_draft/tests
"$RSCRIPT" --vanilla tests/run_tests.r
"$RSCRIPT" --vanilla tests/test_publication_benchmarks.r
POLICY_PYTHON="$POLICY_PYTHON" "$RSCRIPT" --vanilla \
  tests/test_policy_process_isolation.r

"$PYTHON" - <<'PY'
from pathlib import Path
from publication_pipeline_draft.focused_window_training_protocol import (
    validate_protocol,
)

path = Path("publication_pipeline_draft/config/focused_walk_forward_mechanisms_v1.json")
protocol, digest = validate_protocol(path)
assert protocol["confirmatory_claim_permitted"] is False
assert len(protocol["experiments"]) == 3
assert len(protocol["seeds"]) == 5
assert len(protocol["contrasts"]) == 2
assert len(protocol["financial_benchmarks"]) == 6
assert protocol["benchmark_candidate_experiment_id"] == \
    "zero_vine_features_keep_cvar_observation"
assert protocol["crra_gamma"] == 2.0
print("Focused mechanism contract passed:", digest)
PY

echo "Focused walk-forward v1 validation passed."
