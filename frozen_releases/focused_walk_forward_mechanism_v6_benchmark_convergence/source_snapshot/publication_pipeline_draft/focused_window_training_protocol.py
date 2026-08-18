#!/usr/bin/env python3
"""Freeze a small, result-driven external-window mechanism experiment.

The broad external framework compares five RL algorithms.  The consumed causal
analysis found no statistically established TD3 advantage over those controls,
but exposed two much more consequential mechanism questions.  This prospective
development contract therefore spends the remaining compute on three matched
TD3 state representations and never labels the retrospective evidence as fresh
confirmation.
"""

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

from publication_pipeline_draft.publication_research_program import validate_program


class FocusedWindowError(RuntimeError):
    pass


ENV_FIELDS = (
    "RETURNS_DATA_FILE", "RETURNS_DATA_KIND", "RETURNS_DATA_MANIFEST",
    "REF_COL", "VINE_TRUNCATION_LEVEL", "SYNTHETIC_RETURNS_FILE",
    "TRAINING_MARGINALS_FILE", "NN_VINE_MODEL_DIR", "FINETUNE_RETURNS_FILE",
    "RL_ALGORITHM", "POLICY_ENCODER", "VINE_OBSERVATION_MODE",
    "VINE_FEATURE_MODE", "CVAR_OBSERVATION_MODE", "CVAR_REWARD_MODE",
    "PRETRAIN_DATA_MODE", "RUN_FINETUNE", "CHECKPOINT_PREFIX",
    "LR_ACTOR", "LR_CRITIC", "ENTROPY_COEF",
    "PRETRAIN_BEHAVIOR_GATE_MODE",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FocusedWindowError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FocusedWindowError(f"Could not read JSON {path}: {error}") from error


def verify_contents(root: Path) -> None:
    contents = root / "CONTENTS.sha256"
    require(contents.is_file(), f"CONTENTS.sha256 not found: {root}")
    for line in contents.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = root / relative.removeprefix("./")
        require(target.is_file() and sha256(target) == expected,
                f"Checksum mismatch: {target}")


def validate_protocol(protocol_path: Path) -> tuple[dict[str, Any], str]:
    protocol = read_json(protocol_path)
    digest = sha256(protocol_path)
    require(protocol.get("schema_version") == 1,
            "Focused mechanism protocol schema must be 1.")
    require(protocol.get("status") ==
            "prospective_retrospective_development_contract",
            "Focused mechanism protocol has the wrong status.")
    require(protocol.get("evidence_class") == "retrospective_walk_forward" and
            protocol.get("confirmatory_claim_permitted") is False,
            "Focused mechanism protocol cannot authorize confirmation.")
    require(float(protocol.get("crra_gamma", 0)) == 2.0,
            "Focused CRRA gamma must match the frozen economic contract.")
    require("confirmatory" not in str(protocol.get("claim_limit", "")).lower() or
            "cannot restore" in str(protocol.get("claim_limit", "")).lower(),
            "Focused mechanism claim limit is unsafe.")
    seeds = [int(value) for value in protocol.get("seeds", [])]
    require(len(seeds) == len(set(seeds)) == 5,
            "Focused mechanism protocol requires exactly five matched seeds.")
    require(int(protocol.get("minimum_successful_seeds_per_experiment", -1)) == 5,
            "All five focused seeds must be retained.")
    require(protocol.get("financial_benchmarks") == [
        "equal_weight", "shrinkage_mean_variance", "dcc_garch",
        "static_vine", "rolling_vine", "dynamic_nn_vine"],
        "Focused financial benchmark family differs from the frozen design.")
    require(protocol.get("benchmark_candidate_experiment_id") ==
            "zero_vine_features_keep_cvar_observation" and
            protocol.get("benchmark_multiplicity_control") ==
            "holm_across_six_financial_benchmarks",
            "Focused benchmark-comparison contract is invalid.")
    experiments = protocol.get("experiments", [])
    expected = {
        "full_vine_state_and_cvar_observation",
        "zero_vine_features_keep_cvar_observation",
        "zero_vine_features_and_cvar_observation",
    }
    require({item.get("experiment_id") for item in experiments} == expected and
            len(experiments) == 3,
            "Focused mechanism experiment set differs from the frozen design.")
    for item in experiments:
        settings = item.get("settings", {})
        required = {
            "RL_ALGORITHM", "POLICY_ENCODER", "VINE_OBSERVATION_MODE",
            "VINE_FEATURE_MODE", "CVAR_OBSERVATION_MODE", "CVAR_REWARD_MODE",
            "PRETRAIN_DATA_MODE", "RUN_FINETUNE", "CHECKPOINT_PREFIX",
            "LR_ACTOR", "LR_CRITIC", "ENTROPY_COEF",
            "PRETRAIN_BEHAVIOR_GATE_MODE",
        }
        require(set(settings) == required,
                f"Incomplete settings: {item.get('experiment_id')}")
        require(settings["RL_ALGORITHM"] == "td3" and
                settings["POLICY_ENCODER"] == "lstm" and
                settings["PRETRAIN_DATA_MODE"] == "vine_synthetic" and
                settings["RUN_FINETUNE"] == "true" and
                settings["PRETRAIN_BEHAVIOR_GATE_MODE"] == "report_only",
                f"Non-matched focused settings: {item.get('experiment_id')}")
        require(settings["VINE_FEATURE_MODE"] in {"full", "zero"} and
                settings["CVAR_OBSERVATION_MODE"] in {"full", "zero"} and
                settings["VINE_OBSERVATION_MODE"] in {"full", "zero"},
                f"Invalid observation mode: {item.get('experiment_id')}")
    require(protocol.get("failure_policy") ==
            "fail_closed_no_seed_substitution_no_silent_fallback",
            "Focused failure policy was weakened.")
    return protocol, digest


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def materialize(repo_root: Path, program_path: Path, protocol_path: Path,
                window_input: Path, artifact_root: Path,
                output: Path) -> dict[str, Any]:
    require(not output.exists(), f"Output already exists: {output}")
    repo_root = repo_root.resolve()
    input_manifest_path = window_input / "return_input_manifest.json"
    return_file = window_input / "window_daily_log_returns.csv"
    require(input_manifest_path.is_file() and return_file.is_file(),
            "Frozen window input is incomplete.")
    verify_contents(window_input)
    panel = read_json(input_manifest_path)
    require(panel.get("release_status") ==
            "frozen_window_return_input_no_confirmation" and
            panel.get("confirmatory_claim_permitted") is False,
            "Unsupported or confirmatory window input.")
    require(sha256(return_file) == panel.get("return_file_sha256"),
            "Window return file hash mismatch.")

    program = validate_program(program_path)
    protocol, protocol_sha256 = validate_protocol(protocol_path)
    require(panel.get("panel_id") == protocol.get("panel_id"),
            "Panel and focused protocol differ.")
    require(panel.get("evidence_class") == protocol.get("evidence_class"),
            "Window evidence class differs from the focused protocol.")
    # This focused design was created after the consumed causal result and is
    # frozen by its own prospective release. It deliberately reuses the
    # original seven-asset panel for retrospective robustness before the
    # licensed external panel is available.
    design = protocol.get("window_design", {})
    require(int(design.get("minimum_train_months", 0)) == 60 and
            int(design.get("validation_months", 0)) == 24 and
            int(design.get("test_months", 0)) == 24 and
            int(design.get("step_months", 0)) == 24 and
            int(design.get("minimum_windows", 0)) == 2 and
            design.get("allow_overlap_between_test_windows") is False,
            "Focused seven-asset window design differs from its contract.")

    window_id = str(panel["window_id"])
    generated = artifact_root / window_id / "training_data"
    bundle = generated / "synthetic_returns.RData"
    jobs: list[dict[str, Any]] = []
    for experiment in protocol["experiments"]:
        for seed in protocol["seeds"]:
            row: dict[str, Any] = {
                "window_id": window_id,
                "panel_id": panel["panel_id"],
                "evidence_class": panel["evidence_class"],
                "experiment_id": experiment["experiment_id"],
                "experiment_label": experiment["label"],
                "experiment_role": experiment["role"],
                "seed": int(seed),
                "output_dir": (artifact_root / window_id / "focused_policies" /
                               experiment["experiment_id"] /
                               f"seed_{seed}").as_posix(),
                "RETURNS_DATA_FILE": return_file.as_posix(),
                "RETURNS_DATA_KIND": "daily_log_returns",
                "RETURNS_DATA_MANIFEST": input_manifest_path.as_posix(),
                "REF_COL": int(panel["reference_asset_index_1based"]),
                "VINE_TRUNCATION_LEVEL": int(panel["vine_truncation_level"]),
                "SYNTHETIC_RETURNS_FILE": bundle.as_posix(),
                "TRAINING_MARGINALS_FILE": (generated /
                    "training_marginal_results.RData").as_posix(),
                "NN_VINE_MODEL_DIR": (generated / "nn_vine_models").as_posix(),
                "FINETUNE_RETURNS_FILE": (generated /
                    "finetune_returns.qs").as_posix(),
                "focused_protocol_sha256": protocol_sha256,
            }
            row.update(experiment["settings"])
            jobs.append(row)
    require(len(jobs) == 15 and len({(row["experiment_id"], row["seed"])
                                     for row in jobs}) == 15,
            "Focused job matrix must be exactly 3 experiments by 5 seeds.")
    generator_environment = {
        "TRAIN_SEED": int(protocol["seeds"][0]),
        "RETURNS_DATA_FILE": return_file.as_posix(),
        "RETURNS_DATA_KIND": "daily_log_returns",
        "RETURNS_DATA_MANIFEST": input_manifest_path.as_posix(),
        "REF_COL": int(panel["reference_asset_index_1based"]),
        "VINE_TRUNCATION_LEVEL": int(panel["vine_truncation_level"]),
        "SYNTHETIC_RETURNS_FILE": bundle.as_posix(),
        "SYNTHETIC_BUNDLE_MANIFEST": (generated /
            "synthetic_bundle_manifest.json").as_posix(),
        "PRETRAIN_RETURNS_FILE": (generated / "pretrain_returns.qs").as_posix(),
        "FINETUNE_RETURNS_FILE": (generated / "finetune_returns.qs").as_posix(),
        "TRAINING_MARGINALS_FILE": (generated /
            "training_marginal_results.RData").as_posix(),
        "NN_VINE_MODEL_DIR": (generated / "nn_vine_models").as_posix(),
        "SYNTHETIC_DIAGNOSTICS_DIR": (generated /
            "synthetic_diagnostics").as_posix(),
        "SYNTHETIC_DISTRIBUTION_FIGURE": (generated /
            "synthetic_monthly_return_distributions.pdf").as_posix(),
        "IMMUTABLE_SYNTHETIC_OUTPUT": "true",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        jobs_path = temporary / "focused_window_jobs.csv"
        write_csv(jobs_path, jobs)
        (temporary / "generator_environment.json").write_text(
            json.dumps(generator_environment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        shutil.copy2(protocol_path, temporary / "focused_mechanism_protocol.json")
        manifest = {
            "schema_version": 1,
            "release_status": "frozen_focused_window_training_contract",
            "window_id": window_id,
            "panel_id": panel["panel_id"],
            "evidence_class": panel["evidence_class"],
            "asset_count": int(panel["asset_count"]),
            "reference_asset": panel["reference_asset"],
            "reference_asset_index_1based": int(
                panel["reference_asset_index_1based"]),
            "vine_truncation_level": int(panel["vine_truncation_level"]),
            "expected_evaluation_periods": int(
                panel["expected_evaluation_periods"]),
            "confirmatory_claim_permitted": False,
            "experiment_count": 3,
            "seed_count": 5,
            "job_count": 15,
            "program_sha256": program.sha256,
            "focused_protocol_sha256": protocol_sha256,
            "return_input_manifest_sha256": sha256(input_manifest_path),
            "jobs_sha256": sha256(jobs_path),
            "artifact_root": artifact_root.as_posix(),
            "scientific_note": (
                "Prospective retrospective robustness test of the two decisive "
                "policy-visible dependence mechanisms; not fresh confirmation."
            ),
        }
        (temporary / "focused_window_training_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (temporary / "READ_ONLY_CONTRACT.txt").write_text(
            "Do not edit. Retrospective mechanism robustness only.\n",
            encoding="utf-8")
        checksum = [f"{sha256(path)}  {path.name}" for path in
                    sorted(temporary.iterdir()) if path.is_file() and
                    path.name != "CONTENTS.sha256"]
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(checksum) + "\n", encoding="ascii")
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--program", type=Path, default=Path(
        "publication_pipeline_draft/config/publication_research_program_v2.json"))
    parser.add_argument("--protocol", type=Path, default=Path(
        "publication_pipeline_draft/config/focused_walk_forward_mechanisms_v1.json"))
    parser.add_argument("--window-input", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = materialize(args.repo_root, args.program, args.protocol,
                             args.window_input.resolve(), args.artifact_root,
                             args.output)
    except (OSError, ValueError, KeyError, FocusedWindowError) as error:
        print(f"FOCUSED WINDOW CONTRACT FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
