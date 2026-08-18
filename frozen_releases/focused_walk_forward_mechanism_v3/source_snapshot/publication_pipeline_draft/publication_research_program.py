#!/usr/bin/env python3
"""Validate and materialize the publication extension research job plan.

This module does not read return data, train policies, or access a holdout.  It
turns the versioned scientific design into a deterministic job matrix and
fails closed when evidence classes, asset dimensions, algorithms, ablations,
or matched-seed rules are internally inconsistent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class ResearchProgramError(RuntimeError):
    """Raised when the research program would permit an invalid comparison."""


REQUIRED_ALGORITHMS = {"td3", "ddpg", "sac", "ppo", "a2c"}
REQUIRED_FINANCIAL = {
    "equal_weight",
    "minimum_variance",
    "risk_parity",
    "shrinkage_mean_variance",
    "mean_cvar",
    "momentum_tilt",
    "dcc_garch",
}
REQUIRED_ABLATIONS = {
    "zero_vine_features_keep_cvar_observation",
    "keep_vine_features_zero_cvar_observation",
    "zero_vine_features_and_cvar_observation",
    "zero_cvar_reward_keep_state",
    "historical_only_no_synthetic_pretraining",
    "moving_block_bootstrap_pretraining",
    "feedforward_capacity_matched",
    "pretrained_only_no_historical_finetuning",
}
EVIDENCE_CLAIM_LIMITS = {
    "retrospective_walk_forward": "development_and_robustness_only",
    "external_cross_sectional": "external_validity_not_temporally_independent",
    "future_temporal": "eligible_for_confirmatory_inference_after_preregistration",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResearchProgramError(message)


def unique(values: Iterable[str], label: str) -> list[str]:
    result = [str(value) for value in values]
    require(len(result) == len(set(result)), f"{label} contains duplicates.")
    return result


def dense_vine_features(asset_count: int) -> int:
    return 3 * asset_count * (asset_count - 1) // 2


def truncated_vine_features(asset_count: int, trees: int) -> int:
    require(1 <= trees < asset_count, "Truncated vine tree count is invalid.")
    return 3 * sum(asset_count - tree for tree in range(1, trees + 1))


def observation_dimension(asset_count: int, vine_features: int) -> int:
    # wealth + returns + vols + CVaR + prior weights + gross + net + vine
    return 4 + 3 * asset_count + vine_features


@dataclass(frozen=True)
class ValidatedProgram:
    raw: dict[str, Any]
    sha256: str
    panel_dimensions: dict[str, dict[str, int | str]]


def validate_program(path: Path) -> ValidatedProgram:
    raw_bytes = path.read_bytes()
    try:
        program = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise ResearchProgramError(f"Invalid research-program JSON: {error}") from error
    require(program.get("schema_version") == 2, "Research program schema must be 2.")
    require(
        program.get("program_status") == "development_framework_not_frozen",
        "This file is a development framework; frozen protocols require a new release.",
    )
    consumed = program.get("consumed_evidence", [])
    require(len(consumed) >= 2, "Consumed main and ablation evidence must be declared.")
    for item in consumed:
        digest = str(item.get("archive_sha256", ""))
        require(
            len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            f"Consumed evidence {item.get('evidence_id')} has an invalid hash.",
        )

    evidence_classes = program.get("evidence_classes", {})
    require(
        evidence_classes == EVIDENCE_CLAIM_LIMITS,
        "Evidence classes or claim limits were weakened.",
    )
    panels = program.get("panels", [])
    panel_ids = unique((item.get("panel_id", "") for item in panels), "panel_id")
    require(all(panel_ids), "Every panel needs a panel_id.")
    panel_dimensions: dict[str, dict[str, int | str]] = {}
    for panel in panels:
        minimum = int(panel["minimum_assets"])
        maximum = int(panel["maximum_assets"])
        assets = unique(panel.get("asset_order", []), f"{panel['panel_id']} asset order")
        require(1 < minimum <= maximum, f"{panel['panel_id']} asset range is invalid.")
        if assets:
            require(
                minimum <= len(assets) <= maximum,
                f"{panel['panel_id']} fixed asset count is outside its declared range.",
            )
            dimension_count = len(assets)
        else:
            require(
                panel.get("status") == "asset_list_must_be_frozen_before_data_access",
                f"{panel['panel_id']} has no assets but is not explicitly blocked.",
            )
            dimension_count = maximum
        vine = panel["vine_representation"]
        if vine["mode"] == "dense_all_tree_dvine":
            feature_count = dense_vine_features(dimension_count)
        elif vine["mode"] == "truncated_dvine":
            feature_count = truncated_vine_features(
                dimension_count, int(vine["maximum_trees"])
            )
        else:
            raise ResearchProgramError(
                f"Unsupported vine representation for {panel['panel_id']}."
            )
        obs_dim = observation_dimension(dimension_count, feature_count)
        require(
            obs_dim <= int(vine["maximum_observation_dimension"]),
            f"{panel['panel_id']} observation dimension {obs_dim} exceeds its ceiling.",
        )
        panel_dimensions[panel["panel_id"]] = {
            "asset_count_for_dimension_check": dimension_count,
            "vine_feature_count": feature_count,
            "observation_dimension": obs_dim,
            "vine_mode": vine["mode"],
        }

    designs = program.get("window_designs", [])
    design_ids = unique((item.get("design_id", "") for item in designs), "design_id")
    require(all(design_ids), "Every window design needs a design_id.")
    for design in designs:
        require(design["panel_id"] in panel_ids, "Window design references unknown panel.")
        evidence = design["evidence_class"]
        require(evidence in evidence_classes, "Unknown window evidence class.")
        require(
            not bool(design["allow_overlap_between_test_windows"]),
            "Publication windows may not overlap.",
        )
        require(int(design["test_months"]) >= 12, "Test windows are too short.")
        require(int(design["minimum_windows"]) >= 1, "Minimum windows must be positive.")
        if evidence == "future_temporal":
            require(
                str(design.get("earliest_test_start", "")) > "2026-07-06",
                "Future confirmation overlaps the consumed holdout.",
            )
            require(
                int(design["minimum_windows"]) >= 2,
                "Future confirmation requires at least two non-overlapping windows.",
            )
        else:
            require(
                "confirmatory" not in str(design["claim_limit"]).lower(),
                "Non-future evidence cannot make a confirmatory claim.",
            )

    financial = set(unique(program.get("financial_benchmarks", []), "benchmarks"))
    require(REQUIRED_FINANCIAL <= financial, "Required financial benchmarks are missing.")
    algorithms = program.get("rl_algorithms", [])
    algorithm_ids = set(
        unique((item.get("algorithm_id", "") for item in algorithms), "RL algorithms")
    )
    require(REQUIRED_ALGORITHMS <= algorithm_ids, "Required RL algorithms are missing.")
    require(
        all(item.get("encoder") == "lstm" for item in algorithms),
        "Main RL algorithm comparisons must use the same recurrent encoder.",
    )
    rl_protocol = program.get("rl_comparison_protocol", {})
    require(
        rl_protocol.get("observation_reward_cost_constraint_contract")
        == "identical_across_algorithms",
        "RL controls must share observations, rewards, costs, and constraints.",
    )
    require(
        rl_protocol.get("hyperparameter_selection")
        == "preregistered_algorithm_specific_defaults_validated_on_training_prefix_only_before_external_test_freeze"
        and rl_protocol.get("post_test_tuning") == "forbidden",
        "RL hyperparameter selection is not causally locked.",
    )
    require(
        all(bool(rl_protocol.get(field)) for field in
            ("td3", "ddpg", "sac", "ppo", "a2c")),
        "Algorithm-specific update definitions are incomplete.",
    )
    ablations = set(unique(program.get("causal_ablations", []), "ablations"))
    require(REQUIRED_ABLATIONS <= ablations, "Required causal ablations are missing.")

    seed_design = program["seed_design"]
    seeds = [int(seed) for seed in seed_design["seeds"]]
    require(len(seeds) == len(set(seeds)) >= 10, "At least ten distinct seeds are required.")
    require(
        int(seed_design["minimum_successful_seeds"]) == len(seeds),
        "All preregistered seeds must pass; partial seed sets are forbidden.",
    )
    costs = program["cost_sensitivity"]
    require(
        {10, 25, 50} <= set(int(value) for value in costs["transaction_cost_bps_one_way"]),
        "The 10/25/50 bps cost stress is incomplete.",
    )
    inference = program["inference"]
    require(inference["primary_outcome"] == "mean_period_crra_utility_difference",
            "Primary outcome must remain CRRA utility.")
    require(inference["primary_benchmark"] == "equal_weight",
            "Primary benchmark must remain equal weight.")
    require(int(inference["tail_minimum_events"]) >= 20,
            "Tail claims require at least 20 events.")
    return ValidatedProgram(
        raw=program,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        panel_dimensions=panel_dimensions,
    )


def job_rows(validated: ValidatedProgram) -> list[dict[str, Any]]:
    program = validated.raw
    seeds = [int(seed) for seed in program["seed_design"]["seeds"]]
    rows: list[dict[str, Any]] = []
    for design in program["window_designs"]:
        for algorithm in program["rl_algorithms"]:
            for seed in seeds:
                rows.append(
                    {
                        "job_family": "rl_algorithm",
                        "design_id": design["design_id"],
                        "evidence_class": design["evidence_class"],
                        "panel_id": design["panel_id"],
                        "experiment_id": algorithm["algorithm_id"],
                        "algorithm": algorithm["algorithm_id"],
                        "encoder": algorithm["encoder"],
                        "seed": seed,
                        "holdout_access_permitted": design["evidence_class"] == "future_temporal",
                    }
                )
        for ablation in program["causal_ablations"]:
            for seed in seeds:
                rows.append(
                    {
                        "job_family": "causal_ablation",
                        "design_id": design["design_id"],
                        "evidence_class": design["evidence_class"],
                        "panel_id": design["panel_id"],
                        "experiment_id": ablation,
                        "algorithm": "td3",
                        "encoder": "mlp" if ablation == "feedforward_capacity_matched" else "lstm",
                        "seed": seed,
                        "holdout_access_permitted": design["evidence_class"] == "future_temporal",
                    }
                )
    return rows


def write_jobs(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise ResearchProgramError(f"Job manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--program", required=True, type=Path)
    jobs = subparsers.add_parser("jobs")
    jobs.add_argument("--program", required=True, type=Path)
    jobs.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validated = validate_program(args.program)
        result: dict[str, Any] = {
            "program_id": validated.raw["program_id"],
            "program_sha256": validated.sha256,
            "panel_dimensions": validated.panel_dimensions,
            "status": "valid_development_framework",
        }
        if args.command == "jobs":
            rows = job_rows(validated)
            write_jobs(args.output, rows)
            result.update({"job_manifest": str(args.output), "job_count": len(rows)})
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except ResearchProgramError as error:
        print(f"RESEARCH PROGRAM FAILURE: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
