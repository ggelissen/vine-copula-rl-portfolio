#!/usr/bin/env python3
"""Freeze the matched-seed RL training contract for one external window."""

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


class WindowTrainingError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WindowTrainingError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def materialize(
    repo_root: Path,
    program_path: Path,
    window_input: Path,
    artifact_root: Path,
    output: Path,
) -> dict[str, Any]:
    require(not output.exists(), f"Output already exists: {output}")
    repo_root = repo_root.resolve()
    input_manifest_path = window_input / "return_input_manifest.json"
    return_file = window_input / "window_daily_log_returns.csv"
    contents_path = window_input / "CONTENTS.sha256"
    require(all(path.is_file() for path in
                (input_manifest_path, return_file, contents_path)),
            "Frozen window input is incomplete.")
    panel = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    require(panel.get("release_status") ==
            "frozen_window_return_input_no_confirmation",
            "Unsupported return-input release.")
    require(panel.get("confirmatory_claim_permitted") is False,
            "Development training input may not authorize confirmation.")
    require(sha256(return_file) == panel.get("return_file_sha256"),
            "Window return file hash mismatch.")
    for line in contents_path.read_text(encoding="ascii").splitlines():
        if line.strip():
            expected, relative = line.split("  ", 1)
            target = window_input / relative
            require(target.is_file() and sha256(target) == expected,
                    f"Window input checksum mismatch: {target}")

    program = validate_program(program_path)
    panels = {item["panel_id"]: item for item in program.raw["panels"]}
    require(panel["panel_id"] in panels, "Window panel is absent from the program.")
    declared = panels[panel["panel_id"]]
    require(list(panel["asset_order"]) == list(declared.get("asset_order", [])) or
            not declared.get("asset_order"),
            "Window asset order differs from the preregistered program.")
    algorithms = [item["algorithm_id"] for item in program.raw["rl_algorithms"]]
    require(set(algorithms) == {"td3", "ddpg", "sac", "ppo", "a2c"},
            "External RL algorithm family is incomplete.")
    seeds = [int(seed) for seed in program.raw["seed_design"]["seeds"]]
    require(len(seeds) == len(set(seeds)) == 10, "Exactly ten matched seeds are required.")

    window_id = str(panel["window_id"])
    generated = artifact_root / window_id / "training_data"
    bundle = generated / "synthetic_returns.RData"
    prefixes = {
        "td3": "td3_lstm_vine", "ddpg": "ddpg_lstm_vine",
        "sac": "sac_lstm_vine", "ppo": "ppo_lstm_vine",
        "a2c": "a2c_lstm_vine",
    }
    optimizer_defaults = {
        "td3": ("0.00003", "0.0001", "0.005"),
        "ddpg": ("0.00003", "0.0001", "0.005"),
        "sac": ("0.0003", "0.0003", "0.1"),
        "ppo": ("0.0003", "0.001", "0.005"),
        "a2c": ("0.0003", "0.001", "0.005"),
    }
    jobs: list[dict[str, Any]] = []
    for algorithm in algorithms:
        for seed in seeds:
            jobs.append({
                "window_id": window_id,
                "panel_id": panel["panel_id"],
                "evidence_class": panel["evidence_class"],
                "algorithm": algorithm,
                "seed": seed,
                "output_dir": (artifact_root / window_id / "policies" /
                               algorithm / f"seed_{seed}").as_posix(),
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
                "RL_ALGORITHM": algorithm,
                "POLICY_ENCODER": "lstm",
                "VINE_OBSERVATION_MODE": "full",
                "VINE_FEATURE_MODE": "full",
                "CVAR_OBSERVATION_MODE": "full",
                "CVAR_REWARD_MODE": "full",
                "PRETRAIN_DATA_MODE": "vine_synthetic",
                "RUN_FINETUNE": "true",
                "CHECKPOINT_PREFIX": prefixes[algorithm],
                "LR_ACTOR": optimizer_defaults[algorithm][0],
                "LR_CRITIC": optimizer_defaults[algorithm][1],
                "ENTROPY_COEF": optimizer_defaults[algorithm][2],
            })
    generator_environment = {
        "TRAIN_SEED": int(program.raw["seed_design"]["seeds"][0]),
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
        jobs_path = temporary / "window_rl_jobs.csv"
        write_csv(jobs_path, jobs)
        (temporary / "generator_environment.json").write_text(
            json.dumps(generator_environment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "release_status": "frozen_development_window_training_contract",
            "window_id": window_id,
            "panel_id": panel["panel_id"],
            "evidence_class": panel["evidence_class"],
            "confirmatory_claim_permitted": False,
            "asset_count": int(panel["asset_count"]),
            "vine_truncation_level": int(panel["vine_truncation_level"]),
            "algorithm_count": len(algorithms),
            "seed_count": len(seeds),
            "job_count": len(jobs),
            "program_sha256": program.sha256,
            "return_input_manifest_sha256": sha256(input_manifest_path),
            "jobs_sha256": sha256(jobs_path),
            "artifact_root": artifact_root.as_posix(),
            "scientific_note": (
                "Every algorithm receives the same states, data, interactions, "
                "rewards, constraints, costs, and matched seed set."
            ),
        }
        (temporary / "window_training_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "READ_ONLY_CONTRACT.txt").write_text(
            "Do not edit. Retrospective/external development training only.\n",
            encoding="utf-8",
        )
        checksum_lines = [
            f"{sha256(path)}  {path.name}"
            for path in sorted(temporary.iterdir())
            if path.is_file() and path.name != "CONTENTS.sha256"
        ]
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="ascii"
        )
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
    parser.add_argument("--window-input", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = materialize(args.repo_root, args.program, args.window_input,
                             args.artifact_root, args.output)
    except (OSError, ValueError, json.JSONDecodeError, WindowTrainingError) as error:
        print(f"WINDOW TRAINING CONTRACT FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
