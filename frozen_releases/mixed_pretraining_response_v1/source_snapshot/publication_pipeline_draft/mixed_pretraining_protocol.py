#!/usr/bin/env python3
"""Validate, materialize, and freeze the mixed-pretraining experiment."""

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

JOB_ENV_FIELDS = (*BASE_FIELDS, "SYNTHETIC_RETURNS_FILE")
EXPERIMENT = "mixed_100synthetic_61historical_pretrain_plus_historical_finetune"
SOURCES = (
    "run_with_config.r", "evaluate_with_config.r", "config/config.yaml",
    "helper/reproducibility.r", "helper/load_data.r", "helper/time_split.r",
    "helper/marginals.r", "benchmark_models/dynamic_vine_NN.r",
    "rl/rl_environment.r", "rl/train_rl.r", "rl/training_sanity_check.r",
    "rl/evaluate_rl.r", "rl/action_projection.py", "rl/recurrent_baselines.py",
    "rl/policy_inference_server_v2.py", "rl/materialize_mixed_pretraining_bundle.r",
    "publication_pipeline_draft/mixed_pretraining_protocol.py",
    "publication_pipeline_draft/run_mixed_pretraining_sweep.py",
    "publication_pipeline_draft/audit_mixed_pretraining_comparison.py",
    "publication_pipeline_draft/generate_mixed_pretraining_comparison_weights.py",
    "publication_pipeline_draft/analyze_mixed_pretraining_response.py",
    "publication_pipeline_draft/config/mixed_pretraining_response_v1.json",
    "publication_pipeline_draft/config/causal_analysis_contract_v2.json",
    "publication_pipeline_draft/config/evaluation_contract.json",
    "publication_pipeline_draft/tests/test_mixed_pretraining_protocol.py",
    "publication_pipeline_draft/MIXED_PRETRAINING_RESPONSE_V1_RUNBOOK.md",
    "hpc/run_mixed_pretraining_response_v1.sh",
)


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"Mixed-pretraining contract not found: {path}")
    raw = path.read_bytes()
    contract = json.loads(raw)
    require(contract.get("schema_version") == 1, "Schema must be 1.")
    require(contract.get("experiment_id") ==
            "post_holdout_mixed_pretraining_response_v1",
            "Experiment identifier changed.")
    require(contract.get("evidence_class") == "post_holdout_explanatory" and
            contract.get("confirmatory_claim_permitted") is False and
            contract.get("consumed_holdout_reused") is True and
            contract.get("terminal_same_holdout_training") is True and
            contract.get("protocol_deviation_from_prior_stop_rule") is True,
            "Evidence boundary is incomplete.")
    require(bool(contract.get("selection_disclosure")) and
            bool(contract.get("outcome_independent_rule")) and
            "explicit protocol deviation" in
            str(contract.get("prior_stop_rule_disclosure", "")),
            "Outcome-independent reporting was not declared.")
    require("final same-holdout" in str(contract.get("stop_rule", "")),
            "Terminal stop rule changed.")
    require(contract.get("failure_policy") ==
            "fail_closed_no_seed_substitution_no_silent_fallback",
            "Failure policy was weakened.")
    geometry = tuple(int(contract.get(name, -1)) for name in (
        "synthetic_unique_episode_count", "historical_unique_episode_count",
        "mixed_unique_episode_count", "pretrain_episode_presentations",
        "finetune_episodes", "episode_length"))
    require(geometry == (100, 61, 161, 1000, 61, 24),
            "Mixed-curriculum geometry changed.")
    require((int(contract.get("synthetic_episode_presentations", -1)),
             int(contract.get("historical_episode_presentations", -1))) ==
            (621, 379), "Exact mixed presentation counts changed.")
    require(contract.get("presentation_protocol") ==
            "proportional_midpoint_interleave_100synthetic_61historical_1000_v1",
            "Presentation rule changed.")
    seeds = [int(value) for value in contract.get("seeds", [])]
    require(len(seeds) == len(set(seeds)) == 10,
            "Exactly ten matched seeds are required.")
    require(int(contract.get("minimum_successful_seeds_per_experiment", -1)) == 10,
            "All ten mixed policies must pass.")
    base = {str(key): str(value)
            for key, value in contract.get("base_model", {}).items()}
    require(set(base) == set(BASE_FIELDS), "Base-model fields are incomplete.")
    require(base["RL_ALGORITHM"] == "td3" and
            base["POLICY_ENCODER"] == "lstm" and
            base["PRETRAIN_DATA_MODE"] == "mixed_historical_synthetic" and
            base["RUN_FINETUNE"] == "true",
            "Mixed experiment changed the controller family or data mode.")
    require(all(base[name] == "zero" for name in (
        "VINE_OBSERVATION_MODE", "VINE_FEATURE_MODE", "CVAR_OBSERVATION_MODE"))
        and base["CVAR_REWARD_MODE"] == "full",
        "Mixed experiment must use the selected masked architecture.")
    require(base["PRETRAIN_EPISODES"] == "1000" and
            base["PRETRAIN_RANDOM_EXPLORATION_STEPS"] == "1000" and
            base["PRETRAIN_NOISE_DECAY"] == "0.998" and
            base["PRETRAIN_BEHAVIOR_GATE_MODE"] == "report_only",
            "Matched optimization schedule changed.")
    experiments = contract.get("experiments", [])
    require(len(experiments) == 1 and
            experiments[0].get("experiment_id") == EXPERIMENT and
            not experiments[0].get("overrides"),
            "Exactly one unmodified mixed experiment is allowed.")
    arms = contract.get("comparison_arms", [])
    require(len(arms) == 4 and {item.get("arm_id") for item in arms} == {
        "historical_only_training", "synthetic_only_training",
        "mixed_pretraining_plus_historical_finetuning",
        "synthetic_pretraining_plus_historical_finetuning"},
        "The four-arm comparison changed.")
    primary = contract.get("primary_contrasts", [])
    require(len(primary) == 3 and all(item.get("candidate") ==
            "mixed_pretraining_plus_historical_finetuning" for item in primary),
            "Three mixed-versus-control contrasts are required.")
    guardrails = contract.get("economic_guardrails", {})
    require(float(guardrails.get("maximum_mean_monthly_turnover_increase", -1)) ==
            0.10 and
            float(guardrails.get("maximum_mean_gross_exposure_increase", -1)) ==
            0.10, "Economic guardrails changed.")
    return contract, hashlib.sha256(raw).hexdigest()


def validate_bundle(repo: Path, contract: dict[str, Any]) -> dict[str, Any]:
    path = repo / contract["bundle"]
    manifest_path = repo / contract["bundle_manifest"]
    require(path.is_file() and manifest_path.is_file(),
            "Materialized mixed bundle or manifest is missing.")
    rows = read_csv(manifest_path)
    require(len(rows) == 1, "Mixed bundle manifest must have one row.")
    row = rows[0]
    require(sha256(path) == row["sha256"].lower(), "Mixed bundle hash mismatch.")
    require((int(row["synthetic_unique_episode_count"]),
             int(row["historical_unique_episode_count"]),
             int(row["mixed_unique_episode_count"]),
             int(row["mixed_episode_presentations"]),
             int(row["finetune_episodes"])) == (100, 61, 161, 1000, 61),
            "Mixed bundle manifest geometry is wrong.")
    require((int(row["synthetic_episode_presentations"]),
             int(row["historical_episode_presentations"])) == (621, 379),
            "Mixed bundle must contain exactly 621 synthetic and 379 historical presentations.")
    require(row["selection_uses_returns_or_diagnostics"].lower() == "false" and
            row["evaluation_data_accessed"].lower() == "false",
            "Bundle selection accessed outcomes or evaluation data.")
    return {"path": contract["bundle"], "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "synthetic_presentations": int(row["synthetic_episode_presentations"]),
            "historical_presentations": int(row["historical_episode_presentations"])}


def validated_rows(contract_path: Path, repo: Path, output_root: Path,
                   validate_inputs: bool = True
                   ) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
    contract, digest = load_contract(contract_path)
    bundle = validate_bundle(repo, contract) if validate_inputs else None
    base = {str(key): str(value) for key, value in contract["base_model"].items()}
    settings = {**base, "SYNTHETIC_RETURNS_FILE": contract["bundle"]}
    item = contract["experiments"][0]
    rows: list[dict[str, Any]] = []
    for seed in contract["seeds"]:
        row: dict[str, Any] = {
            "job_family": "mixed_pretraining_response_v1",
            "experiment_id": EXPERIMENT, "seed": int(seed),
            "output_dir": (output_root / EXPERIMENT /
                           f"seed_{int(seed)}").as_posix(),
            "scientific_question": item["scientific_question"],
            "contract_sha256": digest,
            "bundle_sha256": bundle["sha256"] if bundle else "",
            "pretrain_episode_presentations": 1000,
        }
        row.update(settings); rows.append(row)
    require(len(rows) == 10, "Job matrix must contain ten mixed policies.")
    return rows, digest, bundle


def write_jobs(path: Path, rows: list[dict[str, Any]], digest: str) -> None:
    require(not path.exists(), f"Job matrix already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n# contract_sha256={digest}\n", encoding="ascii")


def verify_release(release: Path, repo: Path, jobs: Path) -> dict[str, Any]:
    manifest_path = release / "mixed_pretraining_release_manifest.json"
    inventory_path = release / "source_inventory.csv"
    contents = release / "CONTENTS.sha256"
    require(all(path.is_file() for path in (manifest_path, inventory_path, contents)),
            "Frozen mixed-pretraining release is incomplete.")
    for line in contents.read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected, relative = line.split("  ", 1)
            target = release / relative
            require(target.is_file() and sha256(target) == expected,
                    f"Frozen release checksum mismatch: {relative}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("release_status") ==
            "frozen_post_holdout_mixed_pretraining_response_v1" and
            manifest.get("jobs_sha256") == sha256(jobs),
            "Frozen release status/jobs mismatch.")
    for row in read_csv(inventory_path):
        live = repo / row["path"]
        require(live.is_file() and sha256(live) == row["sha256"],
                f"Live source drifted after freeze: {row['path']}")
    return manifest


def freeze(repo: Path, contract_path: Path, jobs: Path, runtime: Path,
           output: Path, archive: Path | None) -> dict[str, Any]:
    require(not output.exists(), f"Mixed-pretraining release exists: {output}")
    rows, contract_sha, bundle = validated_rows(
        contract_path, repo, Path("data/mixed_pretraining_runs_v1"))
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
        shutil.copy2(jobs, temporary / "mixed_pretraining_jobs_v1.csv")
        shutil.copy2(runtime, temporary / "training_python_runtime.json")
        manifest = {
            "schema_version": 1,
            "release_status": "frozen_post_holdout_mixed_pretraining_response_v1",
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "terminal_same_holdout_training": True,
            "job_count": 10, "experiment_count": 1, "seed_count": 10,
            "contract_sha256": contract_sha, "jobs_sha256": sha256(jobs),
            "runtime_sha256": sha256(runtime), "source_count": len(inventory),
            "training_bundle_sha256": bundle["sha256"],
            "synthetic_presentations": bundle["synthetic_presentations"],
            "historical_presentations": bundle["historical_presentations"],
        }
        (temporary / "mixed_pretraining_release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "READ_ONLY_RELEASE.txt").write_text(
            "Immutable post-holdout mixed-pretraining release. Do not edit.\n",
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
        "publication_pipeline_draft/config/mixed_pretraining_response_v1.json"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/mixed_pretraining_runs_v1"))
    parser.add_argument("--jobs", type=Path); parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime", type=Path); parser.add_argument("--bundle", type=Path)
    args = parser.parse_args(); repo = args.repo_root.resolve()
    contract_path = (repo / args.contract).resolve()
    try:
        if args.command == "validate":
            contract, digest = load_contract(contract_path)
            result = {"status": "contract_valid", "contract_sha256": digest,
                      "experiments": 1, "new_policies": 10,
                      "comparison_arms": len(contract["comparison_arms"])}
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
    except (DoseProtocolError, OSError, ValueError, KeyError,
            json.JSONDecodeError) as error:
        print(f"MIXED PRETRAINING PROTOCOL FAILURE: {error}"); return 1
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
