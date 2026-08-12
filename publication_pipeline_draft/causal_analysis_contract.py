#!/usr/bin/env python3
"""Validate and materialize the outcome-blind causal analysis plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from publication_pipeline_draft.causal_ablation_protocol import validated_rows


class CausalAnalysisContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CausalAnalysisContractError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ValidatedContract:
    raw: dict[str, Any]
    sha256: str
    experiment_ids: tuple[str, ...]


def load_contract(path: Path) -> ValidatedContract:
    path = path.resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CausalAnalysisContractError(f"Invalid analysis contract: {error}") from error
    require(int(raw.get("schema_version", 0)) == 1, "Unsupported analysis schema.")
    require(raw.get("analysis_status") == "prospective_before_causal_evaluation",
            "Analysis plan is not prospective.")
    require(raw.get("evidence_class") == "post_holdout_explanatory",
            "Evidence class must be post_holdout_explanatory.")
    require(raw.get("confirmatory_claim_permitted") is False,
            "The consumed holdout cannot authorize confirmatory claims.")
    require(raw.get("training_diagnostics_accessed_before_revision") is True and
            raw.get("causal_evaluation_returns_accessed_before_revision") is False,
            "The v3 training-diagnostic revision disclosure is incomplete.")
    require(raw.get("mixed_revision_carry_forward_permitted") is True,
            "The disclosed 70-v2 plus 60-v3 carry-forward rule is missing.")

    root = path.parents[2]
    training_path = root / str(raw.get("training_contract", ""))
    require(training_path.is_file(), "Referenced causal training contract is missing.")
    expected_digest = str(raw.get("expected_training_contract_sha256", ""))
    require(sha256(training_path) == expected_digest,
            "Causal training contract hash differs from the analysis plan.")
    rows, validated_digest = validated_rows(
        training_path, Path(str(raw.get(
            "training_output_root", "data/publication_extension_runs_v2"))))
    require(validated_digest == expected_digest,
            "Validated causal training contract digest differs.")
    experiments = tuple(sorted({str(row["experiment_id"]) for row in rows}))
    require(len(experiments) == 13, "Exactly thirteen experiments are required.")
    seeds = [int(value) for value in raw.get("expected_seeds", [])]
    require(len(seeds) == len(set(seeds)) == 10, "Exactly ten distinct seeds are required.")
    require({int(row["seed"]) for row in rows} == set(seeds),
            "Analysis seeds differ from the training contract.")
    reference = str(raw.get("reference_experiment_id", ""))
    require(reference in experiments, "Reference experiment is undeclared.")

    primary = raw.get("primary_component_contrasts", [])
    algorithms = raw.get("algorithm_robustness_contrasts", [])
    primary_ids = [str(item.get("alternative_experiment_id", "")) for item in primary]
    algorithm_ids = [str(item.get("alternative_experiment_id", "")) for item in algorithms]
    require(len(primary_ids) == len(set(primary_ids)) == 8,
            "Exactly eight distinct primary component contrasts are required.")
    require(len(algorithm_ids) == len(set(algorithm_ids)) == 4,
            "Exactly four distinct algorithm robustness contrasts are required.")
    require(set(primary_ids + algorithm_ids + [reference]) == set(experiments),
            "Analysis contrasts do not cover exactly the thirteen experiments.")

    inference = raw.get("inference", {})
    require(inference.get("primary_outcome") == "paired_monthly_crra_utility_difference",
            "Primary outcome changed.")
    require(inference.get("bootstrap") == "circular_moving_block",
            "Bootstrap method changed.")
    require(int(inference.get("bootstrap_replications", 0)) >= 9999,
            "At least 9,999 bootstrap replications are required.")
    require(int(inference.get("bootstrap_block_length", 0)) >= 2,
            "Bootstrap block length must preserve serial dependence.")
    require(inference.get("primary_multiplicity") ==
            "holm_across_eight_component_contrasts", "Primary multiplicity changed.")
    economics = raw.get("economics", {})
    require(economics.get("ensemble_construction") ==
            "arithmetic_mean_target_weights_then_rescore_costs",
            "Ensembles must be formed in weight space and rescored.")
    require(economics.get("return_aggregation_forbidden") is True,
            "Averaging seed returns is forbidden.")
    require(economics.get("turnover_convention") == "drifted_pretrade_v1" and
            economics.get("financing_proration") == "actual_calendar_days_v1",
            "Common economic accounting convention changed.")
    for field in ("turnover_cost", "annual_short_borrow_rate",
                  "annual_cash_borrow_rate"):
        require(float(economics.get(field, -1)) >= 0,
                f"Invalid non-negative economics field: {field}")
    require(int(economics.get("day_count_basis", 0)) == 365,
            "Actual-calendar financing must use a 365-day basis.")
    required_columns = raw.get("required_period_columns", [])
    require(len(required_columns) == len(set(required_columns)) and
            {"experiment_id", "strategy_level", "seed", "net_return",
             "decision_date", "holding_end_date"} <= set(required_columns),
            "Required evaluation-period schema is incomplete.")
    return ValidatedContract(raw=raw, sha256=sha256(path), experiment_ids=experiments)


def materialize_plan(contract_path: Path, output: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    require(not output.exists(), f"Analysis plan already exists: {output}")
    reference = contract.raw["reference_experiment_id"]
    rows: list[dict[str, Any]] = []
    for family, items, alternative in (
        ("primary_component", contract.raw["primary_component_contrasts"],
         "reference_greater_than_ablation"),
        ("algorithm_robustness", contract.raw["algorithm_robustness_contrasts"],
         "two_sided_exploratory"),
    ):
        for index, item in enumerate(items, start=1):
            rows.append({
                "contrast_family": family,
                "family_index": index,
                "reference_experiment_id": reference,
                "alternative_experiment_id": item["alternative_experiment_id"],
                "label": item["label"],
                "alternative_hypothesis": alternative,
                "primary_outcome": contract.raw["inference"]["primary_outcome"],
                "economic_effect": contract.raw["inference"]["economic_effect"],
                "multiplicity_method": "holm_within_family",
                "contract_sha256": contract.sha256,
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return {"analysis_id": contract.raw["analysis_id"],
            "contract_sha256": contract.sha256, "contrast_count": len(rows),
            "primary_contrasts": 8, "algorithm_contrasts": 4,
            "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "materialize"))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v1.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            contract = load_contract(args.contract)
            result = {"analysis_id": contract.raw["analysis_id"],
                      "contract_sha256": contract.sha256,
                      "experiment_count": len(contract.experiment_ids),
                      "status": "valid_prospective_analysis_plan"}
        else:
            require(args.output is not None, "--output is required for materialize.")
            result = materialize_plan(args.contract, args.output)
    except (CausalAnalysisContractError, OSError, ValueError) as error:
        print(f"CAUSAL ANALYSIS CONTRACT FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
