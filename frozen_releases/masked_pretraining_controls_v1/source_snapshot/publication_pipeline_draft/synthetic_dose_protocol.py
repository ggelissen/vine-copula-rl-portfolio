#!/usr/bin/env python3
"""Validate, materialize, and freeze the 100-path synthetic-dose experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any


class DoseProtocolError(RuntimeError):
    pass


BASE_FIELDS = (
    "RL_ALGORITHM", "POLICY_ENCODER", "VINE_OBSERVATION_MODE",
    "VINE_FEATURE_MODE",
    "CVAR_OBSERVATION_MODE", "CVAR_REWARD_MODE", "PRETRAIN_DATA_MODE",
    "RUN_FINETUNE", "CHECKPOINT_PREFIX", "LR_ACTOR", "LR_CRITIC",
    "ENTROPY_COEF", "PRETRAIN_BEHAVIOR_GATE_MODE", "PRETRAIN_EPISODES",
    "PRETRAIN_RANDOM_EXPLORATION_STEPS", "PRETRAIN_NOISE_DECAY",
    "PRETRAIN_BEHAVIOR_GATE_WINDOW",
)
JOB_ENV_FIELDS = (*BASE_FIELDS, "SYNTHETIC_RETURNS_FILE")
EXPERIMENTS = {
    "synthetic_100_full_vine_state",
    "synthetic_100_no_policy_visible_dependence",
}
SOURCES = (
    "run_with_config.r", "evaluate_with_config.r", "config/config.yaml",
    "helper/reproducibility.r",
    "helper/load_data.r", "helper/time_split.r", "helper/marginals.r",
    "benchmark_models/dynamic_vine_NN.r", "rl/rl_environment.r",
    "rl/train_rl.r", "rl/training_sanity_check.r", "rl/evaluate_rl.r",
    "rl/action_projection.py", "rl/recurrent_baselines.py",
    "rl/policy_inference_server_v2.py",
    "rl/materialize_synthetic_dose_bundle.r",
    "publication_pipeline_draft/synthetic_dose_protocol.py",
    "publication_pipeline_draft/run_synthetic_dose_sweep.py",
    "publication_pipeline_draft/audit_synthetic_dose_sweep.py",
    "publication_pipeline_draft/generate_synthetic_dose_policy_weights.py",
    "publication_pipeline_draft/analyze_synthetic_dose_response.py",
    "publication_pipeline_draft/publication_pipeline.py",
    "publication_pipeline_draft/analyze_causal_results.py",
    "publication_pipeline_draft/config/synthetic_dose_response_v1.json",
    "publication_pipeline_draft/config/causal_analysis_contract_v2.json",
    "publication_pipeline_draft/config/evaluation_contract.json",
    "publication_pipeline_draft/tests/test_synthetic_dose_protocol.py",
    "publication_pipeline_draft/SYNTHETIC_DOSE_RESPONSE_V1_RUNBOOK.md",
    "hpc/run_synthetic_dose_response_v1.sh",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DoseProtocolError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), f"CSV is empty: {path}")
    return rows


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"Dose contract not found: {path}")
    raw = path.read_bytes()
    try:
        contract = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DoseProtocolError(f"Invalid dose contract: {error}") from error
    require(contract.get("schema_version") == 1, "Dose schema must be 1.")
    require(contract.get("evidence_class") == "post_holdout_explanatory",
            "Dose experiment must remain post-holdout explanatory.")
    require(contract.get("confirmatory_claim_permitted") is False and
            contract.get("consumed_holdout_reused") is True,
            "Consumed-holdout disclosure is missing.")
    require(bool(contract.get("design_revision_reason")) and
            bool(contract.get("selection_disclosure")),
            "Post-focused-analysis design revision is not disclosed.")
    require(contract.get("failure_policy") ==
            "fail_closed_no_seed_substitution_no_silent_fallback",
            "Dose failure policy was weakened.")
    require(int(contract.get("parent_pretrain_episodes", -1)) == 1000 and
            int(contract.get("selected_pretrain_episodes", -1)) == 100 and
            int(contract.get("finetune_episodes", -1)) == 61 and
            int(contract.get("episode_length", -1)) == 24,
            "Dose geometry must remain 1000 -> 100 synthetic and 61 historical.")
    require(contract.get("selection_protocol") ==
            "systematic_midpoint_100_of_1000_v1",
            "The deterministic subset rule changed.")
    seeds = [int(value) for value in contract.get("seeds", [])]
    require(len(seeds) == len(set(seeds)) == 10,
            "Exactly ten distinct matched seeds are required.")
    require(int(contract.get("minimum_successful_seeds_per_experiment", -1)) == 10,
            "All ten seeds must pass per dose experiment.")
    experiments = contract.get("experiments", [])
    identifiers = {str(item.get("experiment_id", "")) for item in experiments}
    require(len(experiments) == 2 and identifiers == EXPERIMENTS,
            "Dose design must contain exactly full-state and no-visible variants.")
    primary = contract.get("primary_contrasts", [])
    require(len(primary) == 3 and
            {(item.get("candidate"), item.get("comparator")) for item in primary} == {
                ("synthetic_100_full_vine_state",
                 "full_vine_state_and_cvar_observation"),
                ("synthetic_100_no_policy_visible_dependence",
                 "zero_vine_features_and_cvar_observation"),
                ("synthetic_100_no_policy_visible_dependence",
                 "synthetic_100_full_vine_state")},
            "The three primary dose/representation contrasts changed.")
    base = {str(key): str(value) for key, value in contract.get("base_model", {}).items()}
    require(set(base) == set(BASE_FIELDS), "Dose base model fields are incomplete.")
    require(base["RL_ALGORITHM"] == "td3" and
            base["POLICY_ENCODER"] == "lstm" and
            base["PRETRAIN_DATA_MODE"] == "vine_synthetic" and
            base["RUN_FINETUNE"] == "true",
            "Dose model family changed.")
    require(base["VINE_OBSERVATION_MODE"] == "full" and
            base["VINE_FEATURE_MODE"] == "full" and
            base["CVAR_OBSERVATION_MODE"] == "full" and
            base["CVAR_REWARD_MODE"] == "full",
            "Dose reference representation changed.")
    require(base["PRETRAIN_BEHAVIOR_GATE_MODE"] == "report_only",
            "Intent-to-train gate mode must be report_only.")
    require(base["PRETRAIN_EPISODES"] == "100" and
            base["PRETRAIN_RANDOM_EXPLORATION_STEPS"] == "100" and
            base["PRETRAIN_BEHAVIOR_GATE_WINDOW"] == "100",
            "The 100-path exposure/warm-up/gate contract changed.")
    require(abs(float(base["PRETRAIN_NOISE_DECAY"]) - 0.998 ** 10) <= 1e-10,
            "Exploration decay must preserve the original normalized schedule.")
    for item in experiments:
        overrides = {str(key): str(value)
                     for key, value in item.get("overrides", {}).items()}
        require(set(overrides) <= set(BASE_FIELDS),
                f"Undeclared override in {item['experiment_id']}.")
        settings = {**base, **overrides}
        if item["experiment_id"] == "synthetic_100_full_vine_state":
            require(not overrides, "Full-state reference may not contain overrides.")
        else:
            require(settings["VINE_OBSERVATION_MODE"] == "zero" and
                    settings["VINE_FEATURE_MODE"] == "zero" and
                    settings["CVAR_OBSERVATION_MODE"] == "zero" and
                    settings["CVAR_REWARD_MODE"] == "full",
                    "No-visible variant must mask policy-visible dependence only.")
    return contract, hashlib.sha256(raw).hexdigest()


def validate_bundle(repo: Path, contract: dict[str, Any]) -> dict[str, str]:
    manifest_path = repo / contract["bundle_manifest"]
    rows = read_csv(manifest_path)
    require(len(rows) == 1, "Dose bundle manifest must contain one row.")
    row = rows[0]
    bundle = repo / contract["bundle"]
    require(bundle.is_file(), f"Dose bundle not found: {bundle}")
    require(row.get("protocol") == contract["selection_protocol"],
            "Bundle selection protocol differs from the contract.")
    require(int(row.get("parent_pretrain_episodes", -1)) == 1000 and
            int(row.get("selected_pretrain_episodes", -1)) == 100 and
            int(row.get("finetune_episodes", -1)) == 61 and
            int(row.get("episode_length", -1)) == 24,
            "Bundle manifest has the wrong episode geometry.")
    require(row.get("evaluation_data_accessed", "").lower() == "false",
            "Dose bundle materialization accessed evaluation data.")
    require(row.get("sha256", "").lower() == sha256(bundle),
            "Dose bundle hash does not match its manifest.")
    indices = [int(value) for value in row.get("selection_indices", "").split(";")]
    expected = [int(((index - 0.5) * 1000) // 100 + 1)
                for index in range(1, 101)]
    require(indices == expected, "Dose bundle uses the wrong deterministic indices.")
    return {**row, "resolved_bundle": str(bundle.resolve()),
            "manifest_sha256": sha256(manifest_path)}


def validated_rows(contract_path: Path, repo: Path,
                   output_root: Path) -> tuple[list[dict[str, Any]], str, dict[str, str]]:
    contract, digest = load_contract(contract_path)
    bundle = validate_bundle(repo, contract)
    base = {str(key): str(value) for key, value in contract["base_model"].items()}
    rows: list[dict[str, Any]] = []
    for item in contract["experiments"]:
        experiment = item["experiment_id"]
        settings = {**base, **{str(key): str(value)
                              for key, value in item.get("overrides", {}).items()}}
        settings["SYNTHETIC_RETURNS_FILE"] = contract["bundle"]
        for seed in contract["seeds"]:
            row: dict[str, Any] = {
                "job_family": "synthetic_dose_response",
                "experiment_id": experiment,
                "seed": int(seed),
                "output_dir": (output_root / experiment /
                               f"seed_{int(seed)}").as_posix(),
                "scientific_question": item["scientific_question"],
                "contract_sha256": digest,
                "bundle_sha256": bundle["sha256"],
            }
            row.update(settings)
            rows.append(row)
    require(len(rows) == 20, "Dose job matrix must contain exactly 20 jobs.")
    return rows, digest, bundle


def write_jobs(path: Path, rows: list[dict[str, Any]], digest: str) -> None:
    require(not path.exists(), f"Job matrix already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
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
    manifest_path = release / "synthetic_dose_release_manifest.json"
    inventory_path = release / "source_inventory.csv"
    contents = release / "CONTENTS.sha256"
    require(manifest_path.is_file() and inventory_path.is_file() and contents.is_file(),
            "Frozen dose release is incomplete.")
    for line in contents.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = release / relative
        require(target.is_file() and sha256(target) == expected,
                f"Frozen dose checksum mismatch: {relative}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("release_status") ==
            "frozen_post_holdout_synthetic_dose_training_v1",
            "Frozen dose release has the wrong status.")
    require(manifest.get("jobs_sha256") == sha256(jobs),
            "Live dose job matrix differs from the frozen release.")
    for row in read_csv(inventory_path):
        live = repo / row["path"]
        require(live.is_file() and sha256(live) == row["sha256"],
                f"Live source drifted after dose freeze: {row['path']}")
    return manifest


def write_archive(source: Path, archive: Path) -> None:
    require(not archive.exists(), f"Archive already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    with tarfile.open(temporary, "w:gz", format=tarfile.PAX_FORMAT) as stream:
        stream.add(source, arcname=source.name)
    os.replace(temporary, archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n", encoding="ascii")


def freeze(repo: Path, contract_path: Path, jobs: Path, runtime: Path,
           output: Path, archive: Path | None) -> dict[str, Any]:
    require(not output.exists(), f"Dose release already exists: {output}")
    rows, contract_digest, bundle = validated_rows(
        contract_path, repo, Path("data/synthetic_dose_response_runs_v1"))
    actual = read_csv(jobs)
    require(actual == [{key: str(value) for key, value in row.items()}
                       for row in rows],
            "Dose job matrix is not the exact validated 20-job design.")
    require(runtime.is_file(), f"Training runtime evidence not found: {runtime}")
    for relative in SOURCES:
        require((repo / relative).is_file(), f"Required dose source missing: {relative}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        snapshot = temporary / "source_snapshot"
        inventory: list[dict[str, Any]] = []
        for relative in SOURCES:
            source = repo / relative
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            inventory.append({"path": relative, "sha256": sha256(destination),
                              "size_bytes": destination.stat().st_size})
        with (temporary / "source_inventory.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
            writer.writeheader(); writer.writerows(inventory)
        shutil.copy2(jobs, temporary / "synthetic_dose_jobs_v1.csv")
        shutil.copy2(repo / json.loads(contract_path.read_text(encoding="utf-8"))[
            "bundle_manifest"], temporary / "synthetic_dose_bundle_manifest.csv")
        shutil.copy2(runtime, temporary / "training_python_runtime.json")
        manifest = {
            "schema_version": 1,
            "release_status": "frozen_post_holdout_synthetic_dose_training_v1",
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "job_count": 20, "experiment_count": 2, "seed_count": 10,
            "contract_sha256": contract_digest,
            "jobs_sha256": sha256(jobs),
            "bundle_sha256": bundle["sha256"],
            "bundle_manifest_sha256": bundle["manifest_sha256"],
            "runtime_sha256": sha256(runtime),
            "source_count": len(inventory),
            "scientific_note": (
                "This post-holdout dose-response experiment may diagnose negative "
                "transfer but cannot create a new confirmatory result on the consumed holdout."),
        }
        (temporary / "synthetic_dose_release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "READ_ONLY_RELEASE.txt").write_text(
            "Immutable post-holdout synthetic-dose training release. Do not edit.\n",
            encoding="utf-8")
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "CONTENTS.sha256").write_text(
            "".join(f"{sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
                    for path in files), encoding="ascii")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if archive is not None:
        write_archive(output, archive)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "materialize-jobs", "freeze"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/synthetic_dose_response_v1.json"))
    parser.add_argument("--output-root", type=Path, default=Path(
        "data/synthetic_dose_response_runs_v1"))
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    contract_path = (repo / args.contract).resolve()
    try:
        if args.command == "validate":
            contract, digest = load_contract(contract_path)
            result = {"status": "contract_valid", "contract_sha256": digest,
                      "experiments": len(contract["experiments"]),
                      "seeds": len(contract["seeds"])}
        elif args.command == "materialize-jobs":
            require(args.output is not None, "--output is required.")
            rows, digest, bundle = validated_rows(
                contract_path, repo, args.output_root)
            write_jobs(args.output, rows, digest)
            result = {"status": "jobs_materialized", "jobs": len(rows),
                      "output": str(args.output), "contract_sha256": digest,
                      "bundle_sha256": bundle["sha256"]}
        else:
            require(all(value is not None for value in
                        (args.jobs, args.output, args.runtime)),
                    "freeze requires --jobs, --runtime, and --output.")
            result = freeze(repo, contract_path, args.jobs.resolve(),
                            args.runtime.resolve(), args.output.resolve(),
                            args.bundle.resolve() if args.bundle else None)
    except (DoseProtocolError, OSError, ValueError, KeyError) as error:
        print(f"SYNTHETIC DOSE PROTOCOL FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
