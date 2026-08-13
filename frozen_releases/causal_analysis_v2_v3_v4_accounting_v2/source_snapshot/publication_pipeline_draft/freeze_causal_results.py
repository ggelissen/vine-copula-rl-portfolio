#!/usr/bin/env python3
"""Freeze a completed causal evaluation, standardized panel, and analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import csv
from pathlib import Path
from typing import Any

from publication_pipeline_draft.causal_analysis_contract import load_contract, require
from publication_pipeline_draft.freeze_causal_analysis_plan import (
    CausalAnalysisFreezeError,
    verify_causal_analysis_release,
    verify_contents,
)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    repo, output = args.repo_root.resolve(), args.output.resolve()
    require(not output.exists(), f"Causal result release already exists: {output}")
    require(args.archive is None or not args.archive.resolve().exists(),
            "Causal result archive already exists.")
    release = verify_causal_analysis_release(args.analysis_release, repo)
    contract = load_contract(args.contract)
    require(release["analysis_contract_sha256"] == contract.sha256,
            "Analysis release and contract differ.")
    verify_contents(args.evaluation_interface)
    verify_contents(args.common_output)
    verify_contents(args.analysis_output)
    verify_contents(args.ensembles)
    bindings = json.loads((args.evaluation_interface /
                           "causal_evaluation_bindings.json").read_text(encoding="utf-8"))
    analysis = json.loads((args.analysis_output /
                           "causal_analysis_manifest.json").read_text(encoding="utf-8"))
    period_manifest_path = args.period_panel.with_suffix(
        args.period_panel.suffix + ".manifest.json")
    panel = json.loads(period_manifest_path.read_text(encoding="utf-8"))
    common_manifest_path = args.common_output / "run_manifest.json"
    common = json.loads(common_manifest_path.read_text(encoding="utf-8"))
    audit_manifest_path = args.audit / "causal_sweep_audit_manifest.json"
    audit = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    merge = json.loads(args.operational_merge_manifest.read_text(encoding="utf-8"))
    policy_manifest_path = args.policy_weights / "causal_policy_weight_manifest.json"
    policy = json.loads(policy_manifest_path.read_text(encoding="utf-8"))
    ensemble_manifest_path = args.ensembles / "causal_ensemble_manifest.json"
    ensemble = json.loads(ensemble_manifest_path.read_text(encoding="utf-8"))
    require(bindings.get("analysis_contract_sha256") == contract.sha256 and
            analysis.get("analysis_contract_sha256") == contract.sha256 and
            panel.get("analysis_contract_sha256") == contract.sha256,
            "A causal artifact uses a different analysis contract.")
    require(args.realized.is_file() and
            bindings.get("realized_panel_sha256") == sha256(args.realized),
            "Realized asset-return panel changed after evaluation materialization.")
    require(analysis.get("status") == "causal_analysis_complete" and
            panel.get("status") == "causal_period_panel_complete",
            "Causal analysis or standardized panel is incomplete.")
    require(sha256(args.period_panel) == analysis.get("period_panel_sha256") ==
            panel.get("causal_period_panel_sha256"),
            "Standardized causal period panel hash differs.")
    require(int(analysis.get("strategy_count", -1)) == 143 and
            int(analysis.get("experiment_count", -1)) == 13,
            "Causal analysis cardinality differs from the contract.")
    expected_locked = int(contract.raw["sample"]["expected_periods"])
    expected_complete = int(contract.raw["sample"].get(
        "observed_complete_periods_before_scoring",
        contract.raw["sample"]["minimum_complete_periods"]))
    require(int(analysis.get("locked_periods_per_strategy", -1)) == expected_locked and
            int(analysis.get("complete_periods_per_strategy", -1)) == expected_complete and
            analysis.get("primary_sample_scope") == "complete_periods",
            "Causal analysis does not preserve the disclosed locked/complete calendar sample.")
    require(int(analysis.get("protocol_eligible_policy_count", -1)) == 130 and
            int(analysis.get("all_economic_diagnostics_pass_count", -1)) == 101 and
            int(analysis.get("report_only_included_count", -1)) == 29,
            "Causal analysis did not preserve the disclosed 101 strict-path / "
            "29 report-only training evidence.")
    require(common.get("contract_sha256") == sha256(
        args.evaluation_interface / "evaluation_contract.json"),
        "Common evaluator used a different accounting contract.")
    require(merge.get("status") ==
            "complete_70_v2_plus_31_v3_plus_29_v4_operational_merge" and
            merge.get("combined_jobs_sha256") == sha256(args.jobs) and
            merge.get("combined_status_sha256") == sha256(args.status),
            "Operational merge is not bound to the supplied 130-job evidence.")
    require(audit.get("status") == "causal_sweep_audit_passed" and
            int(audit.get("job_count", -1)) == 130 and
            audit.get("jobs_sha256") == sha256(args.jobs) and
            audit.get("status_sha256") == sha256(args.status) and
            audit.get("operational_merge_manifest_sha256") == sha256(
                args.operational_merge_manifest),
            "Checkpoint audit is not bound to the operational merge.")
    with (args.audit / "checkpoint_audit.csv").open(
            newline="", encoding="utf-8") as stream:
        checkpoint_audit = list(csv.DictReader(stream))
    require(len(checkpoint_audit) == 130 and
            len({(row["experiment_id"], row["seed"])
                 for row in checkpoint_audit}) == 130,
            "Checkpoint audit does not contain 130 unique policies.")
    require(policy.get("status") == "causal_policy_weight_replay_complete" and
            int(policy.get("policy_count", -1)) == 130 and
            policy.get("analysis_contract_sha256") == contract.sha256 and
            bindings.get("policy_weight_manifest_sha256") == sha256(
                args.policy_weights / "causal_policy_weight_manifest.csv"),
            "Policy-weight evidence is invalid or unbound.")
    with (args.policy_weights / "causal_policy_weight_manifest.csv").open(
            newline="", encoding="utf-8") as stream:
        policy_rows = list(csv.DictReader(stream))
    require(len(policy_rows) == 130, "Policy-weight manifest must contain 130 rows.")
    for row in policy_rows:
        path = (repo / row["path"]).resolve()
        require(path.is_file() and sha256(path) == row["sha256"],
                f"Policy-weight file changed after replay: {path}")
    require(ensemble.get("status") == "causal_weight_ensembles_complete" and
            int(ensemble.get("ensemble_count", -1)) == 13 and
            ensemble.get("analysis_contract_sha256") == contract.sha256 and
            bindings.get("ensemble_weight_manifest_sha256") == sha256(
                args.ensembles / "ensemble_weight_manifest.csv"),
            "Ensemble-weight evidence is invalid or unbound.")
    with (args.ensembles / "ensemble_weight_manifest.csv").open(
            newline="", encoding="utf-8") as stream:
        ensemble_rows = list(csv.DictReader(stream))
    require(len(ensemble_rows) == 13,
            "Ensemble-weight manifest must contain 13 rows.")
    for row in ensemble_rows:
        path = (args.ensembles / row["path"]).resolve()
        require(path.is_file() and sha256(path) == row["sha256"],
                f"Ensemble-weight file changed after construction: {path}")
    require(bindings.get("checkpoint_audit_manifest_sha256") == sha256(
                audit_manifest_path),
            "Evaluation interface is not bound to the checkpoint audit.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.copytree(args.analysis_output, temporary / "analysis_results")
        shutil.copytree(args.analysis_release, temporary / "analysis_plan_release")
        shutil.copytree(args.evaluation_interface, temporary / "evaluation_interface")
        shutil.copytree(args.common_output, temporary / "common_accounting")
        shutil.copytree(args.audit, temporary / "checkpoint_audit")
        shutil.copytree(args.policy_weights, temporary / "policy_weights")
        shutil.copytree(args.ensembles, temporary / "policy_ensembles")
        realized_directory = temporary / "realized_input"
        realized_directory.mkdir()
        shutil.copy2(args.realized,
                     realized_directory / "realized_asset_gross.csv")
        shutil.copy2(args.jobs, temporary / args.jobs.name)
        shutil.copy2(args.status, temporary / args.status.name)
        shutil.copy2(args.operational_merge_manifest,
                     temporary / args.operational_merge_manifest.name)
        shutil.copy2(args.period_panel, temporary / args.period_panel.name)
        shutil.copy2(period_manifest_path, temporary / period_manifest_path.name)
        result = {
            "schema_version": 1,
            "release_status": "frozen_post_holdout_causal_results",
            "analysis_id": contract.raw["analysis_id"],
            "analysis_contract_sha256": contract.sha256,
            "analysis_plan_release_contents_sha256": release["release_contents_sha256"],
            "operational_merge_manifest_sha256": sha256(
                args.operational_merge_manifest),
            "checkpoint_audit_manifest_sha256": sha256(audit_manifest_path),
            "policy_weight_manifest_sha256": sha256(
                args.policy_weights / "causal_policy_weight_manifest.csv"),
            "ensemble_weight_manifest_sha256": sha256(
                args.ensembles / "ensemble_weight_manifest.csv"),
            "causal_period_panel_sha256": sha256(args.period_panel),
            "strategy_count": 143, "experiment_count": 13,
            "primary_contrast_count": 8, "algorithm_contrast_count": 4,
            "protocol_eligible_policy_count": analysis.get(
                "protocol_eligible_policy_count"),
            "all_economic_diagnostics_pass_count": analysis.get(
                "all_economic_diagnostics_pass_count"),
            "report_only_included_count": analysis.get(
                "report_only_included_count"),
            "operational_revision": contract.raw.get("operational_revision"),
            "locked_periods_per_strategy": analysis.get(
                "locked_periods_per_strategy"),
            "complete_periods_per_strategy": analysis.get(
                "complete_periods_per_strategy"),
            "primary_sample_scope": analysis.get("primary_sample_scope"),
            "causal_performance_outcomes_scored_before_revision":
                contract.raw.get(
                    "causal_performance_outcomes_scored_before_revision"),
            "all_preregistered_results_reported": True,
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "claim_limit": "mechanism attribution on a previously consumed holdout",
            "self_contained_result_evidence": True,
            "raw_target_weights_included": True,
            "realized_asset_return_panel_included": True,
            "neural_checkpoints_included": False,
            "checkpoint_integrity_bound_by_sha256_audit": True,
        }
        (temporary / "causal_result_release_manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lines = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "CONTENTS.sha256":
                lines.append(f"{sha256(path)}  {path.relative_to(temporary).as_posix()}")
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(lines) + "\n", encoding="ascii")
        os.replace(temporary, output)
        if args.archive is not None:
            from publication_pipeline_draft.freeze_training_release import deterministic_tar
            args.archive.parent.mkdir(parents=True, exist_ok=True)
            deterministic_tar(output, args.archive.resolve())
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v1.json"))
    parser.add_argument("--analysis-release", required=True, type=Path)
    parser.add_argument("--evaluation-interface", required=True, type=Path)
    parser.add_argument("--common-output", required=True, type=Path)
    parser.add_argument("--period-panel", required=True, type=Path)
    parser.add_argument("--analysis-output", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--operational-merge-manifest", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--policy-weights", required=True, type=Path)
    parser.add_argument("--ensembles", required=True, type=Path)
    parser.add_argument("--realized", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.contract = (args.repo_root / args.contract).resolve()
    for name in ("analysis_release", "evaluation_interface", "common_output",
                 "period_panel", "analysis_output", "jobs", "status",
                 "operational_merge_manifest", "audit", "policy_weights",
                 "ensembles", "realized", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.archive is not None:
        args.archive = args.archive.resolve()
    try:
        result = freeze(args)
    except (CausalAnalysisFreezeError, OSError, ValueError, KeyError,
            json.JSONDecodeError) as error:
        print(f"CAUSAL RESULT FREEZE FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
