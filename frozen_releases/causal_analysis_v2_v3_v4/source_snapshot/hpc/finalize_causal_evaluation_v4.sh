#!/usr/bin/env bash
# Fail-closed, staged finalization of the 70-v2 + 31-v3 + 29-v4 causal study.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/gabirel/miniforge3/bin/python3}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/gabirel/miniforge3/envs/vine-rl/bin/python}"
POLICY_PYTHON="${POLICY_PYTHON:-/gabirel/venvs/copula-eval-torch271-cpu/bin/python}"
RSCRIPT="${RSCRIPT:-/gabirel/miniforge3/bin/Rscript}"
CAUSAL_REPLAY_WORKERS="${CAUSAL_REPLAY_WORKERS:-4}"

export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
cd "$REPO_ROOT"

JOBS_MERGED=protocol_manifests/causal_jobs_v2_v3_v4_merged.csv
STATUS_MERGED=protocol_manifests/causal_sweep_status_v2_v3_v4_merged.csv
MERGE_MANIFEST=protocol_manifests/causal_v2_v3_v4_operational_merge.json
PLAN_RELEASE=frozen_releases/causal_analysis_v2_v3_v4
AUDIT=analysis_outputs/causal_sweep_audit_v2_v3_v4
WEIGHTS=analysis_outputs/causal_policy_weights_v2_v3_v4
ENSEMBLES=analysis_outputs/causal_policy_ensembles_v2_v3_v4
INTERFACE=analysis_outputs/causal_evaluation_interface_v2_v3_v4
ACCOUNTING=analysis_outputs/causal_common_accounting_v2_v3_v4
PANEL=analysis_outputs/causal_strategy_periods_v2_v3_v4.csv
RESULTS=analysis_outputs/causal_analysis_results_v2_v3_v4
RESULT_RELEASE=frozen_releases/causal_results_v2_v3_v4
REALIZED=locked_evaluation/main_oos_v4_operational_retry/inputs/realized_asset_gross.csv

require_file() {
  test -f "$1" || { echo "Missing required file: $1" >&2; exit 1; }
}

require_absent() {
  test ! -e "$1" || {
    echo "Refusing to overwrite immutable output: $1" >&2
    exit 1
  }
}

preflight() {
  "$PYTHON" -m compileall -q publication_pipeline_draft
  "$PYTHON" -m pytest -q publication_pipeline_draft/tests
  "$PYTHON" -m publication_pipeline_draft.causal_analysis_contract validate \
    --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json
  require_file protocol_manifests/causal_sweep_status_v4_retry29.csv
  require_file frozen_releases/publication_extension_v4/CONTENTS.sha256
  (cd frozen_releases/publication_extension_v4 && sha256sum -c CONTENTS.sha256)
}

merge_training_evidence() {
  require_absent "$JOBS_MERGED"
  require_absent "$STATUS_MERGED"
  require_absent "$MERGE_MANIFEST"
  "$PYTHON" -m publication_pipeline_draft.merge_causal_three_revision_retry \
    --repo-root . \
    --v2-jobs protocol_manifests/causal_jobs_v2.csv \
    --v2-status protocol_manifests/causal_sweep_status_v2.csv \
    --v2-release frozen_releases/publication_extension_v2 \
    --v3-jobs protocol_manifests/causal_jobs_v3.csv \
    --v3-status protocol_manifests/causal_sweep_status_v3_retry60.csv \
    --v3-release frozen_releases/publication_extension_v3_retry \
    --v4-jobs protocol_manifests/causal_jobs_v4.csv \
    --v4-status protocol_manifests/causal_sweep_status_v4_retry29.csv \
    --v4-release frozen_releases/publication_extension_v4 \
    --output-jobs "$JOBS_MERGED" \
    --output-status "$STATUS_MERGED" \
    --output-manifest "$MERGE_MANIFEST"
}

freeze_plan() {
  require_file "$MERGE_MANIFEST"
  local contrast_plan=protocol_manifests/causal_contrast_plan_v2_v3_v4.csv
  require_absent "$PLAN_RELEASE"
  require_absent "${PLAN_RELEASE}.tar.gz"
  if test -f "$contrast_plan"; then
    "$PYTHON" -c 'import csv, hashlib, sys; from pathlib import Path; plan=Path(sys.argv[1]); contract=Path(sys.argv[2]); digest=hashlib.sha256(contract.read_bytes()).hexdigest(); rows=list(csv.DictReader(plan.open(encoding="utf-8", newline=""))); assert len(rows)==12, "Existing contrast plan must contain 12 rows"; assert {r["contrast_family"] for r in rows}=={"primary_component", "algorithm_robustness"}; assert all(r["contract_sha256"]==digest for r in rows), "Existing contrast plan uses a different contract"; print(f"Validated resumable contrast plan: {plan} ({len(rows)} contrasts; contract={digest})")' \
      "$contrast_plan" \
      publication_pipeline_draft/config/causal_analysis_contract_v1.json
  else
    "$PYTHON" -m publication_pipeline_draft.causal_analysis_contract materialize \
      --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
      --output "$contrast_plan"
  fi
  "$PYTHON" -m publication_pipeline_draft.freeze_causal_analysis_plan \
    --repo-root . \
    --extension-release frozen_releases/publication_extension_v4 \
    --intermediate-extension-release frozen_releases/publication_extension_v3_retry \
    --carried-extension-release frozen_releases/publication_extension_v2 \
    --operational-merge-manifest "$MERGE_MANIFEST" \
    --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
    --output "$PLAN_RELEASE" \
    --archive "${PLAN_RELEASE}.tar.gz"
  (cd "$PLAN_RELEASE" && sha256sum -c CONTENTS.sha256)
  (cd frozen_releases && sha256sum -c causal_analysis_v2_v3_v4.tar.gz.sha256)
}

audit_checkpoints() {
  require_file "$PLAN_RELEASE/causal_analysis_release_manifest.json"
  require_absent "$AUDIT"
  "$TRAIN_PYTHON" -m publication_pipeline_draft.audit_causal_sweep \
    --jobs "$JOBS_MERGED" \
    --status "$STATUS_MERGED" \
    --operational-merge-manifest "$MERGE_MANIFEST" \
    --repo-root . \
    --output "$AUDIT"
}

replay_policies() {
  require_file "$AUDIT/causal_sweep_audit_manifest.json"
  require_absent "$WEIGHTS"
  "$PYTHON" -m publication_pipeline_draft.generate_causal_policy_weights \
    --repo-root . \
    --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
    --analysis-release "$PLAN_RELEASE" \
    --jobs "$JOBS_MERGED" \
    --audit "$AUDIT" \
    --config config/config.yaml \
    --policy-python "$POLICY_PYTHON" \
    --rscript "$RSCRIPT" \
    --workers "$CAUSAL_REPLAY_WORKERS" \
    --output "$WEIGHTS"
}

assemble_ensembles() {
  require_file "$WEIGHTS/causal_policy_weight_manifest.csv"
  require_absent "$ENSEMBLES"
  "$PYTHON" -m publication_pipeline_draft.assemble_causal_policy_ensembles \
    --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
    --weight-manifest "$WEIGHTS/causal_policy_weight_manifest.csv" \
    --repo-root . \
    --output "$ENSEMBLES"
  (cd "$ENSEMBLES" && sha256sum -c CONTENTS.sha256)
}

common_accounting() {
  require_file "$REALIZED"
  require_file "$AUDIT/checkpoint_audit.csv"
  require_absent "$INTERFACE"
  require_absent "$ACCOUNTING"
  "$PYTHON" -m publication_pipeline_draft.materialize_causal_evaluation \
    --repo-root . \
    --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
    --analysis-release "$PLAN_RELEASE" \
    --policy-weights "$WEIGHTS" \
    --ensembles "$ENSEMBLES" \
    --audit "$AUDIT" \
    --realized "$REALIZED" \
    --output "$INTERFACE"
  "$PYTHON" -m publication_pipeline_draft.causal_common_accounting \
    --contract "$INTERFACE/evaluation_contract.json" \
    --realized "$REALIZED" \
    --strategies "$INTERFACE/strategy_manifest.csv" \
    --output "$ACCOUNTING"
  (cd "$INTERFACE" && sha256sum -c CONTENTS.sha256)
  (cd "$ACCOUNTING" && sha256sum -c CONTENTS.sha256)
}

analyze_results() {
  require_file "$ACCOUNTING/run_manifest.json"
  require_absent "$PANEL"
  require_absent "$RESULTS"
  "$PYTHON" -m publication_pipeline_draft.export_causal_period_panel \
    --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
    --common-output "$ACCOUNTING" \
    --output "$PANEL"
  "$PYTHON" -m publication_pipeline_draft.analyze_causal_results \
    --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
    --period-panel "$PANEL" \
    --output "$RESULTS"
  (cd "$RESULTS" && sha256sum -c CONTENTS.sha256)
}

freeze_results() {
  require_file "$RESULTS/causal_analysis_manifest.json"
  require_absent "$RESULT_RELEASE"
  require_absent "${RESULT_RELEASE}.tar.gz"
  "$PYTHON" -m publication_pipeline_draft.freeze_causal_results \
    --repo-root . \
    --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
    --analysis-release "$PLAN_RELEASE" \
    --evaluation-interface "$INTERFACE" \
    --common-output "$ACCOUNTING" \
    --period-panel "$PANEL" \
    --analysis-output "$RESULTS" \
    --jobs "$JOBS_MERGED" \
    --status "$STATUS_MERGED" \
    --operational-merge-manifest "$MERGE_MANIFEST" \
    --audit "$AUDIT" \
    --policy-weights "$WEIGHTS" \
    --ensembles "$ENSEMBLES" \
    --realized "$REALIZED" \
    --output "$RESULT_RELEASE" \
    --archive "${RESULT_RELEASE}.tar.gz"
  (cd "$RESULT_RELEASE" && sha256sum -c CONTENTS.sha256)
  (cd frozen_releases && sha256sum -c causal_results_v2_v3_v4.tar.gz.sha256)
}

case "${1:-}" in
  preflight) preflight ;;
  merge) merge_training_evidence ;;
  freeze-plan) freeze_plan ;;
  audit) audit_checkpoints ;;
  replay) replay_policies ;;
  ensembles) assemble_ensembles ;;
  accounting) common_accounting ;;
  analyze) analyze_results ;;
  freeze-results) freeze_results ;;
  all)
    preflight
    merge_training_evidence
    freeze_plan
    audit_checkpoints
    replay_policies
    assemble_ensembles
    common_accounting
    analyze_results
    freeze_results
    ;;
  *)
    echo "Usage: $0 {preflight|merge|freeze-plan|audit|replay|ensembles|accounting|analyze|freeze-results|all}" >&2
    exit 2
    ;;
esac
