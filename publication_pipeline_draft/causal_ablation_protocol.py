#!/usr/bin/env python3
"""Validate and materialize the matched-seed causal ablation job matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class AblationProtocolError(RuntimeError):
    pass


ALLOWED_MODES = {"full", "zero"}
ALLOWED_ALGORITHMS = {"td3", "ddpg", "sac", "ppo", "a2c"}
ALLOWED_ENCODERS = {"lstm", "mlp"}
ALLOWED_PRETRAIN = {
    "vine_synthetic", "historical_prefix_repeated", "moving_block_bootstrap"
}
REQUIRED_EXPERIMENTS = {
    "full_vine_state_and_cvar_observation",
    "zero_vine_features_keep_cvar_observation",
    "keep_vine_features_zero_cvar_observation",
    "zero_vine_features_and_cvar_observation",
    "zero_cvar_reward_keep_state",
    "historical_only_no_synthetic_pretraining",
    "moving_block_bootstrap_pretraining",
    "feedforward_capacity_matched",
    "pretrained_only_no_historical_finetuning",
}
ENV_FIELDS = (
    "RL_ALGORITHM", "POLICY_ENCODER", "VINE_FEATURE_MODE",
    "CVAR_OBSERVATION_MODE", "CVAR_REWARD_MODE", "PRETRAIN_DATA_MODE",
    "RUN_FINETUNE", "SYNTHETIC_RETURNS_FILE", "CHECKPOINT_PREFIX",
    "LR_ACTOR", "LR_CRITIC", "ENTROPY_COEF",
    "PRETRAIN_BEHAVIOR_GATE_MODE",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AblationProtocolError(message)


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        contract = json.loads(raw)
    except json.JSONDecodeError as error:
        raise AblationProtocolError(f"Invalid ablation JSON: {error}") from error
    require(contract.get("schema_version") == 2, "Ablation schema must be 2.")
    require(contract.get("applies_to_consumed_main_holdout") is False,
            "Causal development experiments cannot overwrite the consumed result.")
    require(contract.get("failure_policy") ==
            "fail_closed_no_seed_substitution_no_silent_fallback",
            "Ablation failure policy was weakened.")
    return contract, hashlib.sha256(raw).hexdigest()


def validate_settings(settings: dict[str, str], label: str) -> None:
    require(set(settings) == set(ENV_FIELDS), f"{label} has incomplete settings.")
    require(settings["RL_ALGORITHM"] in ALLOWED_ALGORITHMS,
            f"{label} has invalid RL algorithm.")
    require(settings["POLICY_ENCODER"] in ALLOWED_ENCODERS,
            f"{label} has invalid encoder.")
    for field in ("VINE_FEATURE_MODE", "CVAR_OBSERVATION_MODE", "CVAR_REWARD_MODE"):
        require(settings[field] in ALLOWED_MODES, f"{label} has invalid {field}.")
    require(settings["PRETRAIN_DATA_MODE"] in ALLOWED_PRETRAIN,
            f"{label} has invalid pretraining source.")
    require(settings["RUN_FINETUNE"] in {"true", "false"},
            f"{label} has invalid fine-tuning switch.")
    require(settings["PRETRAIN_BEHAVIOR_GATE_MODE"] in {"strict", "report_only"},
            f"{label} has invalid behavior-gate mode.")
    require(settings["SYNTHETIC_RETURNS_FILE"].endswith(".RData"),
            f"{label} bundle must be an RData file.")
    require(settings["CHECKPOINT_PREFIX"].replace("_", "").isalnum(),
            f"{label} checkpoint prefix is unsafe.")
    for field in ("LR_ACTOR", "LR_CRITIC", "ENTROPY_COEF"):
        try:
            value = float(settings[field])
        except ValueError as error:
            raise AblationProtocolError(f"{label} has non-numeric {field}.") from error
        require(value > 0, f"{label} has non-positive {field}.")


def validated_rows(contract_path: Path, output_root: Path) -> tuple[list[dict[str, Any]], str]:
    contract, digest = load_contract(contract_path)
    base = {str(key): str(value) for key, value in contract["base_model"].items()}
    validate_settings(base, "base model")
    experiments = contract.get("experiments", [])
    identifiers = [str(item.get("experiment_id", "")) for item in experiments]
    require(len(identifiers) == len(set(identifiers)), "Ablation IDs are duplicated.")
    require(set(identifiers) == REQUIRED_EXPERIMENTS,
            "The causal ablation set is incomplete or contains undeclared tests.")
    seeds = [int(seed) for seed in contract.get("seeds", [])]
    require(len(seeds) == len(set(seeds)) == 10, "Exactly ten matched seeds are required.")
    require(int(contract["minimum_successful_seeds_per_experiment"]) == len(seeds),
            "Every preregistered seed must pass.")
    required_metadata = set(contract.get("checkpoint_metadata_required", []))
    require(required_metadata == {
        "rl_algorithm", "policy_encoder", "vine_feature_mode",
        "cvar_observation_mode", "cvar_reward_mode", "pretrain_data_mode",
        "run_finetune", "pretrain_behavior_gate_mode", "checkpoint_schema"},
        "Checkpoint metadata contract is incomplete.")

    rows: list[dict[str, Any]] = []
    for item in experiments:
        experiment_id = str(item["experiment_id"])
        overrides = {str(key): str(value) for key, value in item.get("overrides", {}).items()}
        require(set(overrides) <= set(ENV_FIELDS),
                f"{experiment_id} overrides an undeclared field.")
        settings = {**base, **overrides}
        validate_settings(settings, experiment_id)
        if experiment_id == "zero_vine_features_and_cvar_observation":
            require(settings["VINE_FEATURE_MODE"] == settings["CVAR_OBSERVATION_MODE"] == "zero",
                    "Joint signal ablation must zero both policy-visible signals.")
        elif experiment_id != "full_vine_state_and_cvar_observation":
            require(bool(overrides), f"{experiment_id} does not alter the base model.")
        for seed in seeds:
            row: dict[str, Any] = {
                "job_family": "causal_ablation",
                "experiment_id": experiment_id,
                "seed": seed,
                "output_dir": (output_root / experiment_id / f"seed_{seed}").as_posix(),
                "scientific_question": item["scientific_question"],
                "contract_sha256": digest,
            }
            row.update(settings)
            rows.append(row)

    for item in contract.get("rl_algorithm_controls", []):
        algorithm = str(item["algorithm_id"])
        require(algorithm in ALLOWED_ALGORITHMS - {"td3"},
                "Only preregistered recurrent algorithm controls may be added.")
        settings = {**base, **{str(k): str(v) for k, v in item["overrides"].items()}}
        validate_settings(settings, f"algorithm {algorithm}")
        require(settings["RL_ALGORITHM"] == algorithm,
                "Algorithm control label and runtime setting disagree.")
        for seed in seeds:
            row = {
                "job_family": "rl_algorithm_control",
                "experiment_id": f"{algorithm}_lstm_full",
                "seed": seed,
                "output_dir": (output_root / f"{algorithm}_lstm_full" /
                               f"seed_{seed}").as_posix(),
                "scientific_question": f"Matched-seed {algorithm.upper()} control.",
                "contract_sha256": digest,
            }
            row.update(settings)
            rows.append(row)
    return rows, digest


def write_matrix(path: Path, rows: list[dict[str, Any]], digest: str) -> None:
    if path.exists():
        raise AblationProtocolError(f"Job matrix already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        f"# contract_sha256={digest}\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_ablation_contract_v2.json"))
    parser.add_argument("--output-root", type=Path, default=Path(
        "data/publication_extension_runs_v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        rows, digest = validated_rows(args.contract, args.output_root)
        write_matrix(args.output, rows, digest)
    except (AblationProtocolError, OSError) as error:
        print(f"ABLATION PROTOCOL FAILURE: {error}")
        return 1
    print(json.dumps({"status": "materialized", "jobs": len(rows),
                      "experiments": len({row['experiment_id'] for row in rows}),
                      "contract_sha256": digest,
                      "output": str(args.output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
