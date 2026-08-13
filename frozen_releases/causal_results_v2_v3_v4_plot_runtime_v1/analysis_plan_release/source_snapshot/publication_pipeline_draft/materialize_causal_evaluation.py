#!/usr/bin/env python3
"""Build the immutable common-accounting interface for 143 causal paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from publication_pipeline_draft.causal_analysis_contract import load_contract, require
from publication_pipeline_draft.freeze_causal_analysis_plan import (
    CausalAnalysisFreezeError,
    verify_causal_analysis_release,
)


class CausalEvaluationError(CausalAnalysisFreezeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def relative(path: Path, destination: Path) -> str:
    return Path(os.path.relpath(path.resolve(), destination.resolve())).as_posix()


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    repo, output = args.repo_root.resolve(), args.output.resolve()
    require(not output.exists(), f"Evaluation interface already exists: {output}")
    release = verify_causal_analysis_release(args.analysis_release, repo)
    contract = load_contract(args.contract)
    require(release["analysis_contract_sha256"] == contract.sha256,
            "Frozen analysis release and contract differ.")
    individual = read_csv(args.policy_weights / "causal_policy_weight_manifest.csv")
    ensemble = read_csv(args.ensembles / "ensemble_weight_manifest.csv")
    audit = read_csv(args.audit / "checkpoint_audit.csv")
    weight_manifest = json.loads((args.policy_weights /
        "causal_policy_weight_manifest.json").read_text(encoding="utf-8"))
    ensemble_manifest = json.loads((args.ensembles /
        "causal_ensemble_manifest.json").read_text(encoding="utf-8"))
    audit_manifest = json.loads((args.audit /
        "causal_sweep_audit_manifest.json").read_text(encoding="utf-8"))
    expected = {(experiment, seed) for experiment in contract.experiment_ids
                for seed in contract.raw["expected_seeds"]}
    individual_by_key = {(row["experiment_id"], int(row["seed"])): row
                         for row in individual}
    audit_by_key = {(row["experiment_id"], int(row["seed"])): row for row in audit}
    require(set(individual_by_key) == set(audit_by_key) == expected,
            "Individual weights/audit are not the exact 130-policy design.")
    require(weight_manifest.get("status") == "causal_policy_weight_replay_complete" and
            int(weight_manifest.get("policy_count", -1)) == 130 and
            weight_manifest.get("analysis_contract_sha256") == contract.sha256 and
            weight_manifest.get("weight_manifest_sha256") == sha256(
                args.policy_weights / "causal_policy_weight_manifest.csv"),
            "Policy-weight replay manifest is invalid or unbound.")
    require(ensemble_manifest.get("status") == "causal_weight_ensembles_complete" and
            int(ensemble_manifest.get("ensemble_count", -1)) == 13 and
            ensemble_manifest.get("analysis_contract_sha256") == contract.sha256 and
            ensemble_manifest.get("individual_weight_manifest_sha256") == sha256(
                args.policy_weights / "causal_policy_weight_manifest.csv"),
            "Ensemble manifest is invalid or unbound to the 130 policies.")
    require(audit_manifest.get("status") == "causal_sweep_audit_passed" and
            int(audit_manifest.get("job_count", -1)) == 130 and
            audit_manifest.get("all_checkpoint_tensors_finite") is True and
            audit_manifest.get("all_behavior_gate_enforcement_valid") is True and
            audit_manifest.get("all_checkpoint_metadata_match") is True,
            "Checkpoint audit manifest is invalid.")
    ensemble_by_experiment = {row["experiment_id"]: row for row in ensemble}
    require(set(ensemble_by_experiment) == set(contract.experiment_ids) and
            len(ensemble) == 13, "Exactly thirteen experiment ensembles are required.")
    require(args.realized.is_file(), f"Realized panel not found: {args.realized}")
    realized_rows = read_csv(args.realized)
    sample = contract.raw["sample"]
    require(len(realized_rows) == int(sample["expected_periods"]),
            "Realized panel does not contain the exact locked period count.")
    require({row["window_id"] for row in realized_rows} == {sample["window_id"]},
            "Realized panel has an unexpected window identifier.")
    complete_values = [str(row["is_complete_period"]).strip().lower()
                       for row in realized_rows]
    require(set(complete_values) <= {"true", "false", "1", "0", "yes", "no"},
            "Realized panel has invalid complete-period flags.")
    observed_complete = sum(value in {"true", "1", "yes"}
                            for value in complete_values)
    declared_complete = int(sample.get(
        "observed_complete_periods_before_scoring", observed_complete))
    require(observed_complete == declared_complete and
            observed_complete >= int(sample["minimum_complete_periods"]),
            "Realized panel completeness differs from the disclosed sample contract.")

    final_root = output
    rows: list[dict[str, Any]] = []
    code_hash = release["release_contents_sha256"]
    config_hash = contract.raw["expected_training_contract_sha256"]
    reference = contract.raw["reference_experiment_id"]
    joint = "zero_vine_features_and_cvar_observation"
    for key in sorted(expected):
        experiment, seed = key
        policy, checkpoint = individual_by_key[key], audit_by_key[key]
        weight_path = (repo / policy["path"]).resolve()
        checkpoint_path = (repo / checkpoint["checkpoint"]).resolve()
        require(weight_path.is_file() and sha256(weight_path) == policy["sha256"],
                f"Individual weight hash mismatch: {key}")
        require(checkpoint_path.is_file() and sha256(checkpoint_path) ==
                checkpoint["sha256"], f"Checkpoint hash mismatch: {key}")
        rows.append({
            "strategy_id": f"{experiment}__seed_{seed}",
            "label": f"{experiment} seed {seed}", "method": experiment,
            "seed": seed, "role": "proposed" if experiment == reference else "ablation",
            # ``gate_pass`` is the legacy common-evaluator eligibility flag.  A
            # report-only causal control remains eligible when every diagnostic
            # is finite and the hard exposure/position checks pass.  Preserve
            # the stricter economic diagnostic result in separate fields so it
            # cannot be mistaken for a universal behavioural-gate pass.
            "completed": True, "gate_pass": True, "ensemble_group": "",
            "include_main": False, "include_inference": False,
            "report_seed_distribution": True,
            "weight_log_path": relative(weight_path, final_root),
            "weight_log_sha256": policy["sha256"],
            "checkpoint_path": relative(checkpoint_path, final_root),
            "checkpoint_sha256": checkpoint["sha256"],
            "config_sha256": config_hash, "code_sha256": code_hash,
            "train_seconds": "", "evaluation_seconds": "",
            "notes": (
                "Matched training seed; not an independent market path; "
                "gate_pass denotes fail-closed protocol eligibility, while "
                "behavior_gate_pass reports every economic diagnostic"
            ),
            "experiment_id": experiment, "strategy_level": "seed",
            "protocol_eligibility_pass": True,
            "behavior_gate_pass": checkpoint["behavior_gate_pass"],
            "behavior_gate_mode": checkpoint["behavior_gate_mode"],
            "behavior_gate_failed_metrics": checkpoint[
                "behavior_gate_failed_metrics"],
            "operational_source": checkpoint["operational_source"],
        })
    for experiment in sorted(contract.experiment_ids):
        item = ensemble_by_experiment[experiment]
        weight_path = (args.ensembles / item["path"]).resolve()
        require(weight_path.is_file() and sha256(weight_path) == item["sha256"],
                f"Ensemble weight hash mismatch: {experiment}")
        rows.append({
            "strategy_id": f"{experiment}_ensemble",
            "label": f"{experiment} weight-space ensemble", "method": experiment,
            "seed": "", "role": "post_holdout_explanatory_ensemble",
            "completed": True, "gate_pass": True, "ensemble_group": "",
            "include_main": True,
            "include_inference": experiment in {reference, joint},
            "report_seed_distribution": False,
            "weight_log_path": relative(weight_path, final_root),
            "weight_log_sha256": item["sha256"],
            "checkpoint_path": "not_applicable", "checkpoint_sha256": "not_applicable",
            "config_sha256": config_hash, "code_sha256": code_hash,
            "train_seconds": "", "evaluation_seconds": "",
            "notes": "Arithmetic mean target weights; costs must be rescored",
            "experiment_id": experiment, "strategy_level": "ensemble",
            "protocol_eligibility_pass": True,
            "behavior_gate_pass": "",
            "behavior_gate_mode": "ensemble_of_audited_members",
            "behavior_gate_failed_metrics": "",
            "operational_source": "weight_space_ensemble",
        })
    require(len(rows) == 143, "Expected 130 individual paths and 13 ensembles.")

    economics = contract.raw["economics"]
    evaluation_contract = {
        "schema_version": 1,
        "evaluation_id": contract.raw["analysis_id"],
        "evidence_class": contract.raw["evidence_class"],
        "confirmatory_claim_permitted": False,
        "expected_locked_periods_per_window": contract.raw["sample"][
            "expected_periods"],
        "minimum_complete_periods_per_window": contract.raw["sample"][
            "minimum_complete_periods"],
        "primary_sample_scope": contract.raw["sample"].get(
            "primary_sample_scope", "complete_periods"),
        "observed_complete_periods": observed_complete,
        "periods_per_year": contract.raw["sample"]["periods_per_year"],
        "annualization_convention": contract.raw["sample"]["annualization_convention"],
        "initial_wealth": economics["initial_wealth"],
        "net_exposure": economics["net_exposure"],
        "gross_leverage": economics["gross_leverage"],
        "max_long_weight": economics["max_long_weight"],
        "max_short_weight": economics["max_short_weight"],
        "turnover_convention": economics["turnover_convention"],
        "turnover_cost": economics["turnover_cost"],
        "financing_proration": economics["financing_proration"],
        "day_count_basis": economics["day_count_basis"],
        "annual_short_borrow_rate": economics["annual_short_borrow_rate"],
        "annual_cash_borrow_rate": economics["annual_cash_borrow_rate"],
        "annual_risk_free_rate": economics["annual_risk_free_rate"],
        "crra_gamma": economics["crra_gamma"],
        "primary_benchmark_id": f"{joint}_ensemble",
        "primary_strategy_id": f"{reference}_ensemble",
        "primary_superiority_test": "one_sided_paired_moving_block_bootstrap_crra",
        "primary_superiority_alpha": 0.05,
        "secondary_multiplicity_control": "holm_within_primary_vs_alternative_family",
        "bootstrap_replications": contract.raw["inference"]["bootstrap_replications"],
        "bootstrap_block_length": contract.raw["inference"]["bootstrap_block_length"],
        "inference_seed": contract.raw["inference"]["inference_seed"],
        "weight_tolerance": economics["weight_tolerance"],
        "require_weight_log_hashes": True,
        "require_checkpoint_hash_for_trained_models": True,
        "require_code_and_config_hashes": True,
        "predeclared_ensembles": [],
        "transaction_cost_sensitivity_bps": [0, 10, 25, 50],
        "annual_short_borrow_sensitivity_percent": [0, 3, 6, 10],
        "figure_strategy_ids": [
            f"{reference}_ensemble", f"{joint}_ensemble",
            "zero_vine_features_keep_cvar_observation_ensemble",
            "keep_vine_features_zero_cvar_observation_ensemble",
            "zero_cvar_reward_keep_state_ensemble",
        ],
        "scientific_note": (
            "Common accounting is an interface step. Preregistered causal inference "
            "is performed by analyze_causal_results.py and remains post-holdout explanatory."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temporary / "evaluation_contract.json").write_text(
            json.dumps(evaluation_contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        with (temporary / "strategy_manifest.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        bindings = {
            "schema_version": 1, "status": "causal_evaluation_interface_complete",
            "analysis_contract_sha256": contract.sha256,
            "analysis_release_contents_sha256": release["release_contents_sha256"],
            "realized_panel_path": str(args.realized.resolve()),
            "realized_panel_sha256": sha256(args.realized),
            "policy_weight_manifest_sha256": sha256(
                args.policy_weights / "causal_policy_weight_manifest.csv"),
            "ensemble_weight_manifest_sha256": sha256(
                args.ensembles / "ensemble_weight_manifest.csv"),
            "checkpoint_audit_manifest_sha256": sha256(
                args.audit / "causal_sweep_audit_manifest.json"),
            "strategy_count": 143, "seed_strategy_count": 130,
            "ensemble_strategy_count": 13,
            "protocol_eligible_policy_count": len(audit),
            "all_economic_diagnostics_pass_count": sum(
                str(row["behavior_gate_pass"]).strip().lower() in
                {"1", "true", "yes"} for row in audit),
            "report_only_included_count": sum(
                row["behavior_gate_mode"] == "report_only" for row in audit),
            "common_realized_returns_required": True,
            "ensemble_returns_averaged": False,
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
        }
        (temporary / "causal_evaluation_bindings.json").write_text(
            json.dumps(bindings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lines = [f"{sha256(path)}  {path.name}" for path in sorted(temporary.iterdir())
                 if path.is_file() and path.name != "CONTENTS.sha256"]
        (temporary / "CONTENTS.sha256").write_text("\n".join(lines) + "\n",
                                                    encoding="ascii")
        os.replace(temporary, output)
        return bindings
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v1.json"))
    parser.add_argument("--analysis-release", required=True, type=Path)
    parser.add_argument("--policy-weights", required=True, type=Path)
    parser.add_argument("--ensembles", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--realized", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.contract = (args.repo_root / args.contract).resolve()
    args.analysis_release = args.analysis_release.resolve()
    args.policy_weights = args.policy_weights.resolve(); args.ensembles = args.ensembles.resolve()
    args.audit = args.audit.resolve(); args.realized = args.realized.resolve()
    args.output = args.output.resolve()
    try:
        result = materialize(args)
    except (CausalAnalysisFreezeError, OSError, ValueError, KeyError,
            json.JSONDecodeError) as error:
        print(f"CAUSAL EVALUATION MATERIALIZATION FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
