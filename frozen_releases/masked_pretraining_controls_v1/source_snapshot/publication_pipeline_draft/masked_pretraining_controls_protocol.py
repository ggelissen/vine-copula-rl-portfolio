#!/usr/bin/env python3
"""Validate, materialize, and freeze the terminal masked pretraining controls."""

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

from publication_pipeline_draft.synthetic_dose_protocol import (
    BASE_FIELDS, DoseProtocolError, read_csv, require, sha256, write_archive,
)


class MaskedControlProtocolError(DoseProtocolError):
    pass


JOB_ENV_FIELDS = (*BASE_FIELDS, "SYNTHETIC_RETURNS_FILE")
EXPERIMENTS = {
    "masked_historical_prefix_1000_presentations",
    "masked_moving_block_bootstrap_1000_presentations",
}
EXPECTED_BUNDLE_HASHES = {
    "historical_prefix_repeated":
        "1f07f655064ccb33dac3c60b2d1ca16ad2c91c73509313de46caeaa15b99e52e",
    "moving_block_bootstrap":
        "0f82cd46391b4c7e08e34470826fef3886cb14a8f3cbf67966a70af36a6bf2a9",
}
SOURCES = (
    "run_with_config.r", "evaluate_with_config.r", "config/config.yaml",
    "helper/reproducibility.r", "helper/load_data.r", "helper/time_split.r",
    "helper/marginals.r", "benchmark_models/dynamic_vine_NN.r",
    "rl/rl_environment.r", "rl/train_rl.r", "rl/training_sanity_check.r",
    "rl/evaluate_rl.r", "rl/action_projection.py", "rl/recurrent_baselines.py",
    "rl/policy_inference_server_v2.py",
    "rl/generate_ablation_training_bundles.r",
    "publication_pipeline_draft/masked_pretraining_controls_protocol.py",
    "publication_pipeline_draft/run_masked_pretraining_controls.py",
    "publication_pipeline_draft/audit_masked_pretraining_controls.py",
    "publication_pipeline_draft/generate_masked_pretraining_control_weights.py",
    "publication_pipeline_draft/analyze_masked_pretraining_controls.py",
    "publication_pipeline_draft/synthetic_dose_protocol.py",
    "publication_pipeline_draft/analyze_synthetic_dose_response.py",
    "publication_pipeline_draft/analyze_synthetic_presentation_response.py",
    "publication_pipeline_draft/publication_pipeline.py",
    "publication_pipeline_draft/analyze_causal_results.py",
    "publication_pipeline_draft/config/masked_pretraining_controls_v1.json",
    "publication_pipeline_draft/config/causal_analysis_contract_v2.json",
    "publication_pipeline_draft/config/evaluation_contract.json",
    "publication_pipeline_draft/tests/test_masked_pretraining_controls_protocol.py",
    "publication_pipeline_draft/MASKED_PRETRAINING_CONTROLS_V1_RUNBOOK.md",
    "publication_pipeline_draft/COMPUTATIONAL_HYPOTHESIS_AUDIT.md",
    "hpc/run_masked_pretraining_controls_v1.sh",
)


def _is_digest(value: object) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef"
                                   for character in text)


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"Masked-control contract not found: {path}")
    raw = path.read_bytes()
    try:
        contract = json.loads(raw)
    except json.JSONDecodeError as error:
        raise MaskedControlProtocolError(
            f"Invalid masked-control contract: {error}") from error
    require(contract.get("schema_version") == 1, "Schema must be 1.")
    require(contract.get("experiment_id") ==
            "terminal_masked_pretraining_controls_v1",
            "Experiment identifier changed.")
    require(contract.get("evidence_class") == "post_holdout_explanatory" and
            contract.get("confirmatory_claim_permitted") is False and
            contract.get("consumed_holdout_reused") is True and
            contract.get("terminal_hpc_experiment") is True,
            "Evidence/terminal-compute disclosure is incomplete.")
    require(bool(contract.get("design_revision_reason")) and
            bool(contract.get("selection_disclosure")),
            "The post-selection design history is not disclosed.")
    require(contract.get("failure_policy") ==
            "fail_closed_no_seed_substitution_no_silent_fallback",
            "Failure policy was weakened.")
    require("no further same-holdout neural training" in
            str(contract.get("stop_rule", "")), "Terminal stop rule changed.")
    require((int(contract.get("pretrain_episode_presentations", -1)),
             int(contract.get("finetune_episodes", -1)),
             int(contract.get("episode_length", -1))) == (1000, 61, 24),
            "Training geometry must remain 1000 pretraining plus 61 fine-tuning episodes.")
    require(contract.get("candidate_experiment_id") ==
            "synthetic_100_unique_1000_presentations_no_policy_visible_dependence",
            "The frozen candidate changed.")
    require(_is_digest(contract.get("candidate_weight_manifest_sha256")),
            "Candidate weight manifest hash is invalid.")

    seeds = [int(value) for value in contract.get("seeds", [])]
    require(len(seeds) == len(set(seeds)) == 10,
            "Exactly ten distinct matched seeds are required.")
    require(int(contract.get("minimum_successful_seeds_per_experiment", -1)) == 10,
            "All ten seeds must pass for both controls.")

    base = {str(key): str(value)
            for key, value in contract.get("base_model", {}).items()}
    require(set(base) == set(BASE_FIELDS), "Base-model fields are incomplete.")
    require(base["RL_ALGORITHM"] == "td3" and
            base["POLICY_ENCODER"] == "lstm" and
            base["RUN_FINETUNE"] == "true" and
            base["PRETRAIN_DATA_MODE"] == "historical_prefix_repeated",
            "Control model family changed.")
    require(all(base[field] == "zero" for field in (
        "VINE_OBSERVATION_MODE", "VINE_FEATURE_MODE",
        "CVAR_OBSERVATION_MODE")) and base["CVAR_REWARD_MODE"] == "full",
            "Controls must use the candidate's no-policy-visible-dependence architecture.")
    require(base["PRETRAIN_EPISODES"] == "1000" and
            base["PRETRAIN_RANDOM_EXPLORATION_STEPS"] == "1000" and
            base["PRETRAIN_NOISE_DECAY"] == "0.998" and
            base["PRETRAIN_BEHAVIOR_GATE_WINDOW"] == "100" and
            base["PRETRAIN_BEHAVIOR_GATE_MODE"] == "report_only",
            "The candidate's 1000-presentation optimization schedule changed.")

    experiments = contract.get("experiments", [])
    identifiers = {str(item.get("experiment_id", "")) for item in experiments}
    require(len(experiments) == 2 and identifiers == EXPERIMENTS,
            "Exactly the two registered masked controls are required.")
    for item in experiments:
        require(_is_digest(item.get("bundle_sha256")),
                f"Invalid bundle hash for {item.get('experiment_id')}.")
        overrides = {str(key): str(value)
                     for key, value in item.get("overrides", {}).items()}
        require(set(overrides) <= set(BASE_FIELDS), "Control has undeclared override.")
        settings = {**base, **overrides}
        require(all(settings[field] == "zero" for field in (
            "VINE_OBSERVATION_MODE", "VINE_FEATURE_MODE",
            "CVAR_OBSERVATION_MODE")) and settings["CVAR_REWARD_MODE"] == "full",
                "A control changes the masked architecture.")
        if item["experiment_id"].startswith("masked_historical"):
            require(settings["PRETRAIN_DATA_MODE"] == "historical_prefix_repeated" and
                    item["bundle_sha256"] ==
                    EXPECTED_BUNDLE_HASHES["historical_prefix_repeated"],
                    "Historical-prefix control changed.")
        else:
            require(settings["PRETRAIN_DATA_MODE"] == "moving_block_bootstrap" and
                    item["bundle_sha256"] ==
                    EXPECTED_BUNDLE_HASHES["moving_block_bootstrap"],
                    "Moving-block control changed.")

    primary = {(item.get("candidate"), item.get("comparator"))
               for item in contract.get("primary_contrasts", [])}
    expected_primary = {
        (contract["candidate_experiment_id"],
         "masked_historical_prefix_1000_presentations"),
        (contract["candidate_experiment_id"],
         "masked_moving_block_bootstrap_1000_presentations"),
    }
    require(primary == expected_primary and len(contract["primary_contrasts"]) == 2,
            "The two terminal generator-value contrasts changed.")
    return contract, hashlib.sha256(raw).hexdigest()


def validate_bundles(repo: Path, contract: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in contract["experiments"]:
        path = repo / item["bundle"]
        require(path.is_file(), f"Control bundle not found: {path}")
        actual = sha256(path)
        require(actual == item["bundle_sha256"].lower(),
                f"Control bundle differs from frozen causal input: {path}")
        records.append({"experiment_id": item["experiment_id"],
                        "path": item["bundle"], "sha256": actual,
                        "size_bytes": path.stat().st_size})
    return records


def validated_rows(contract_path: Path, repo: Path, output_root: Path,
                   validate_inputs: bool = True
                   ) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    contract, digest = load_contract(contract_path)
    bundles = validate_bundles(repo, contract) if validate_inputs else []
    base = {str(key): str(value) for key, value in contract["base_model"].items()}
    rows: list[dict[str, Any]] = []
    for item in contract["experiments"]:
        settings = {**base, **{str(key): str(value)
                              for key, value in item.get("overrides", {}).items()}}
        settings["SYNTHETIC_RETURNS_FILE"] = item["bundle"]
        for seed in contract["seeds"]:
            row: dict[str, Any] = {
                "job_family": "terminal_masked_pretraining_control",
                "experiment_id": item["experiment_id"], "seed": int(seed),
                "output_dir": (output_root / item["experiment_id"] /
                               f"seed_{int(seed)}").as_posix(),
                "scientific_question": item["scientific_question"],
                "contract_sha256": digest,
                "bundle_sha256": item["bundle_sha256"],
                "pretrain_episode_presentations": 1000,
            }
            row.update(settings); rows.append(row)
    require(len(rows) == 20, "Job matrix must contain exactly 20 jobs.")
    return rows, digest, bundles


def write_jobs(path: Path, rows: list[dict[str, Any]], digest: str) -> None:
    require(not path.exists(), f"Job matrix already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor); temporary = Path(name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n# contract_sha256={digest}\n",
        encoding="ascii")


def verify_release(release: Path, repo: Path, jobs: Path) -> dict[str, Any]:
    manifest_path = release / "masked_pretraining_control_release_manifest.json"
    inventory_path = release / "source_inventory.csv"
    contents = release / "CONTENTS.sha256"
    require(all(path.is_file() for path in (manifest_path, inventory_path, contents)),
            "Frozen masked-control release is incomplete.")
    for line in contents.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = release / relative
        require(target.is_file() and sha256(target) == expected,
                f"Frozen masked-control checksum mismatch: {relative}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("release_status") ==
            "frozen_terminal_masked_pretraining_controls_v1",
            "Frozen release has the wrong status.")
    require(manifest.get("jobs_sha256") == sha256(jobs),
            "Live jobs differ from the frozen release.")
    for row in read_csv(inventory_path):
        live = repo / row["path"]
        require(live.is_file() and sha256(live) == row["sha256"],
                f"Live source drifted after masked-control freeze: {row['path']}")
    return manifest


def freeze(repo: Path, contract_path: Path, jobs: Path, runtime: Path,
           output: Path, archive: Path | None) -> dict[str, Any]:
    require(not output.exists(), f"Masked-control release exists: {output}")
    rows, contract_sha, bundles = validated_rows(
        contract_path, repo, Path("data/masked_pretraining_control_runs_v1"))
    actual = read_csv(jobs)
    require(actual == [{key: str(value) for key, value in row.items()}
                       for row in rows], "Job matrix is not the validated design.")
    require(runtime.is_file(), f"Training runtime evidence not found: {runtime}")
    for relative in SOURCES:
        require((repo / relative).is_file(), f"Required source missing: {relative}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        snapshot = temporary / "source_snapshot"
        inventory: list[dict[str, Any]] = []
        for relative in SOURCES:
            source, destination = repo / relative, snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            inventory.append({"path": relative, "sha256": sha256(destination),
                              "size_bytes": destination.stat().st_size})
        with (temporary / "source_inventory.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
            writer.writeheader(); writer.writerows(inventory)
        shutil.copy2(jobs, temporary / "masked_pretraining_control_jobs_v1.csv")
        shutil.copy2(runtime, temporary / "training_python_runtime.json")
        with (temporary / "training_bundle_inventory.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(bundles[0]))
            writer.writeheader(); writer.writerows(bundles)
        manifest = {
            "schema_version": 1,
            "release_status": "frozen_terminal_masked_pretraining_controls_v1",
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "terminal_hpc_experiment": True,
            "job_count": 20, "experiment_count": 2, "seed_count": 10,
            "pretrain_episode_presentations": 1000,
            "historical_finetune_episode_count": 61,
            "contract_sha256": contract_sha, "jobs_sha256": sha256(jobs),
            "runtime_sha256": sha256(runtime), "source_count": len(inventory),
            "training_bundle_hashes": {
                row["experiment_id"]: row["sha256"] for row in bundles},
            "scientific_note": (
                "Terminal same-holdout training closes the pretraining-source "
                "confound within the selected masked architecture; it is explanatory."),
        }
        (temporary / "masked_pretraining_control_release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "READ_ONLY_RELEASE.txt").write_text(
            "Immutable terminal post-holdout masked-control release. Do not edit.\n",
            encoding="utf-8")
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "CONTENTS.sha256").write_text(
            "".join(f"{sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
                    for path in files), encoding="ascii")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise
    if archive is not None:
        write_archive(output, archive)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "materialize-jobs", "freeze"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/masked_pretraining_controls_v1.json"))
    parser.add_argument("--output-root", type=Path, default=Path(
        "data/masked_pretraining_control_runs_v1"))
    parser.add_argument("--jobs", type=Path); parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime", type=Path); parser.add_argument("--bundle", type=Path)
    args = parser.parse_args(); repo = args.repo_root.resolve()
    contract_path = (repo / args.contract).resolve()
    try:
        if args.command == "validate":
            contract, digest = load_contract(contract_path)
            result = {"status": "contract_valid", "contract_sha256": digest,
                      "experiments": len(contract["experiments"]),
                      "seeds": len(contract["seeds"]),
                      "new_policies": 20, "terminal_hpc_experiment": True}
        elif args.command == "materialize-jobs":
            require(args.output is not None, "--output is required.")
            rows, digest, bundles = validated_rows(
                contract_path, repo, args.output_root)
            write_jobs(args.output, rows, digest)
            result = {"status": "jobs_materialized", "jobs": len(rows),
                      "output": str(args.output), "contract_sha256": digest,
                      "bundle_hashes": {row["experiment_id"]: row["sha256"]
                                        for row in bundles}}
        else:
            require(all(value is not None for value in
                        (args.jobs, args.output, args.runtime)),
                    "freeze requires --jobs, --runtime, and --output.")
            result = freeze(repo, contract_path, args.jobs.resolve(),
                            args.runtime.resolve(), args.output.resolve(),
                            args.bundle.resolve() if args.bundle else None)
    except (DoseProtocolError, OSError, ValueError, KeyError) as error:
        print(f"MASKED PRETRAINING CONTROL PROTOCOL FAILURE: {error}"); return 1
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
