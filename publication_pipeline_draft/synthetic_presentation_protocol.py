#!/usr/bin/env python3
"""Validate and freeze the 100-unique/1000-presentation experiment."""

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


class PresentationProtocolError(DoseProtocolError):
    pass


JOB_ENV_FIELDS = (*BASE_FIELDS, "SYNTHETIC_RETURNS_FILE")
EXPERIMENTS = {
    "synthetic_100_unique_1000_presentations_full_vine_state",
    "synthetic_100_unique_1000_presentations_no_policy_visible_dependence",
}
SOURCES = (
    "run_with_config.r", "evaluate_with_config.r", "config/config.yaml",
    "helper/reproducibility.r", "helper/load_data.r", "helper/time_split.r",
    "helper/marginals.r", "benchmark_models/dynamic_vine_NN.r",
    "rl/rl_environment.r", "rl/train_rl.r", "rl/training_sanity_check.r",
    "rl/evaluate_rl.r", "rl/action_projection.py", "rl/recurrent_baselines.py",
    "rl/policy_inference_server_v2.py",
    "rl/materialize_synthetic_presentation_bundle.r",
    "publication_pipeline_draft/synthetic_presentation_protocol.py",
    "publication_pipeline_draft/run_synthetic_presentation_sweep.py",
    "publication_pipeline_draft/audit_synthetic_presentation_sweep.py",
    "publication_pipeline_draft/generate_synthetic_presentation_policy_weights.py",
    "publication_pipeline_draft/analyze_synthetic_presentation_response.py",
    "publication_pipeline_draft/synthetic_dose_protocol.py",
    "publication_pipeline_draft/analyze_synthetic_dose_response.py",
    "publication_pipeline_draft/publication_pipeline.py",
    "publication_pipeline_draft/analyze_causal_results.py",
    "publication_pipeline_draft/config/synthetic_presentation_response_v2.json",
    "publication_pipeline_draft/config/causal_analysis_contract_v2.json",
    "publication_pipeline_draft/config/evaluation_contract.json",
    "publication_pipeline_draft/tests/test_synthetic_presentation_protocol.py",
    "publication_pipeline_draft/SYNTHETIC_PRESENTATION_RESPONSE_V2_RUNBOOK.md",
    "hpc/run_synthetic_presentation_response_v2.sh",
)


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"Presentation contract not found: {path}")
    raw = path.read_bytes()
    try:
        contract = json.loads(raw)
    except json.JSONDecodeError as error:
        raise PresentationProtocolError(
            f"Invalid presentation contract: {error}") from error
    require(contract.get("schema_version") == 1, "Schema must be 1.")
    require(contract.get("evidence_class") == "post_holdout_explanatory" and
            contract.get("confirmatory_claim_permitted") is False and
            contract.get("consumed_holdout_reused") is True,
            "The consumed-holdout evidence disclosure is incomplete.")
    require(bool(contract.get("design_revision_reason")) and
            bool(contract.get("selection_disclosure")),
            "The data-dependent design history is not disclosed.")
    require(contract.get("failure_policy") ==
            "fail_closed_no_seed_substitution_no_silent_fallback",
            "Failure policy was weakened.")
    geometry = (
        int(contract.get("parent_pretrain_episodes", -1)),
        int(contract.get("synthetic_unique_episode_count", -1)),
        int(contract.get("synthetic_episode_presentations", -1)),
        int(contract.get("repetition_count", -1)),
        int(contract.get("finetune_episodes", -1)),
        int(contract.get("episode_length", -1)),
    )
    require(geometry == (1000, 100, 1000, 10, 61, 24),
            "Presentation geometry must be 100 unique x 10 passes plus 61 historical.")
    require(contract.get("presentation_protocol") ==
            "ordered_10_passes_of_systematic_midpoint_100_v2",
            "The deterministic presentation rule changed.")
    for field in ("source_100_path_bundle_sha256",
                  "dose100_weight_manifest_sha256"):
        value = str(contract.get(field, "")).lower()
        require(len(value) == 64 and all(character in "0123456789abcdef"
                                         for character in value),
                f"Contract field {field} is not a SHA-256 digest.")
    seeds = [int(value) for value in contract.get("seeds", [])]
    require(len(seeds) == len(set(seeds)) == 10,
            "Exactly ten distinct matched seeds are required.")
    require(int(contract.get("minimum_successful_seeds_per_experiment", -1)) == 10,
            "All ten seeds must pass for both policies.")
    experiments = contract.get("experiments", [])
    identifiers = {str(item.get("experiment_id", "")) for item in experiments}
    require(len(experiments) == 2 and identifiers == EXPERIMENTS,
            "The design must contain exactly the two registered policies.")
    base = {str(key): str(value)
            for key, value in contract.get("base_model", {}).items()}
    require(set(base) == set(BASE_FIELDS), "Base-model fields are incomplete.")
    require(base["RL_ALGORITHM"] == "td3" and
            base["POLICY_ENCODER"] == "lstm" and
            base["PRETRAIN_DATA_MODE"] == "vine_synthetic" and
            base["RUN_FINETUNE"] == "true",
            "Model family changed.")
    require(base["VINE_OBSERVATION_MODE"] == "full" and
            base["VINE_FEATURE_MODE"] == "full" and
            base["CVAR_OBSERVATION_MODE"] == "full" and
            base["CVAR_REWARD_MODE"] == "full",
            "Full-state reference changed.")
    require(base["PRETRAIN_EPISODES"] == "1000" and
            base["PRETRAIN_RANDOM_EXPLORATION_STEPS"] == "1000" and
            base["PRETRAIN_BEHAVIOR_GATE_WINDOW"] == "100" and
            abs(float(base["PRETRAIN_NOISE_DECAY"]) - 0.998) <= 1e-12,
            "Original 1000-presentation exploration/update schedule changed.")
    require(base["PRETRAIN_BEHAVIOR_GATE_MODE"] == "report_only",
            "Intent-to-train gate mode must remain report_only.")
    for item in experiments:
        overrides = {str(key): str(value)
                     for key, value in item.get("overrides", {}).items()}
        require(set(overrides) <= set(BASE_FIELDS),
                f"Undeclared override in {item['experiment_id']}.")
        settings = {**base, **overrides}
        if item["experiment_id"].endswith("full_vine_state"):
            require(not overrides, "Full-state reference may not contain overrides.")
        else:
            require(settings["VINE_OBSERVATION_MODE"] == "zero" and
                    settings["VINE_FEATURE_MODE"] == "zero" and
                    settings["CVAR_OBSERVATION_MODE"] == "zero" and
                    settings["CVAR_REWARD_MODE"] == "full",
                    "No-visible policy must mask only policy-visible dependence.")
    primary = {(item.get("candidate"), item.get("comparator"))
               for item in contract.get("primary_contrasts", [])}
    expected_primary = {
        ("synthetic_100_unique_1000_presentations_full_vine_state",
         "synthetic_100_full_vine_state"),
        ("synthetic_100_unique_1000_presentations_full_vine_state",
         "full_vine_state_and_cvar_observation"),
        ("synthetic_100_unique_1000_presentations_no_policy_visible_dependence",
         "synthetic_100_no_policy_visible_dependence"),
        ("synthetic_100_unique_1000_presentations_no_policy_visible_dependence",
         "zero_vine_features_and_cvar_observation"),
    }
    require(primary == expected_primary and len(contract["primary_contrasts"]) == 4,
            "The four mechanism contrasts changed.")
    return contract, hashlib.sha256(raw).hexdigest()


def validate_bundle(repo: Path, contract: dict[str, Any]) -> dict[str, str]:
    manifest_path = repo / contract["bundle_manifest"]
    rows = read_csv(manifest_path)
    require(len(rows) == 1, "Presentation bundle manifest must contain one row.")
    row = rows[0]
    bundle = repo / contract["bundle"]
    source = repo / contract["source_100_path_bundle"]
    require(bundle.is_file() and source.is_file(),
            "Presentation bundle or exact 100-path source is missing.")
    require(row.get("protocol") == contract["presentation_protocol"],
            "Bundle presentation protocol differs from the contract.")
    require(int(row.get("parent_pretrain_episodes", -1)) == 1000 and
            int(row.get("synthetic_unique_episode_count", -1)) == 100 and
            int(row.get("synthetic_episode_presentations", -1)) == 1000 and
            int(row.get("repetition_count", -1)) == 10 and
            int(row.get("finetune_episodes", -1)) == 61 and
            int(row.get("episode_length", -1)) == 24,
            "Bundle manifest has the wrong training geometry.")
    require(row.get("presentation_rule") == "ten_ordered_complete_passes" and
            row.get("selection_uses_returns_or_diagnostics", "").lower() == "false" and
            row.get("evaluation_data_accessed", "").lower() == "false",
            "Bundle construction is not deterministic and evaluation-blind.")
    require(row.get("sha256", "").lower() == sha256(bundle),
            "Presentation bundle hash differs from its manifest.")
    require(row.get("source_100_path_sha256", "").lower() == sha256(source),
            "The exact frozen 100-path source has changed.")
    require(sha256(source) == contract["source_100_path_bundle_sha256"].lower(),
            "The 100-path source differs from the completed frozen v1 experiment.")
    indices = [int(value) for value in
               row.get("source_indices_per_pass", "").split(";")]
    require(indices == list(range(1, 101)),
            "Each ordered pass must contain source indices 1 through 100 once.")
    return {**row, "resolved_bundle": str(bundle.resolve()),
            "manifest_sha256": sha256(manifest_path)}


def validated_rows(contract_path: Path, repo: Path,
                   output_root: Path) -> tuple[list[dict[str, Any]], str, dict[str, str]]:
    contract, digest = load_contract(contract_path)
    bundle = validate_bundle(repo, contract)
    base = {str(key): str(value) for key, value in contract["base_model"].items()}
    rows: list[dict[str, Any]] = []
    for item in contract["experiments"]:
        settings = {**base, **{str(key): str(value)
                              for key, value in item.get("overrides", {}).items()}}
        settings["SYNTHETIC_RETURNS_FILE"] = contract["bundle"]
        for seed in contract["seeds"]:
            row: dict[str, Any] = {
                "job_family": "synthetic_presentation_response",
                "experiment_id": item["experiment_id"], "seed": int(seed),
                "output_dir": (output_root / item["experiment_id"] /
                               f"seed_{int(seed)}").as_posix(),
                "scientific_question": item["scientific_question"],
                "contract_sha256": digest, "bundle_sha256": bundle["sha256"],
                "synthetic_unique_episode_count": 100,
                "synthetic_episode_presentations": 1000,
            }
            row.update(settings)
            rows.append(row)
    require(len(rows) == 20, "Job matrix must contain exactly 20 jobs.")
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
    manifest_path = release / "synthetic_presentation_release_manifest.json"
    inventory_path = release / "source_inventory.csv"
    contents = release / "CONTENTS.sha256"
    require(manifest_path.is_file() and inventory_path.is_file() and contents.is_file(),
            "Frozen presentation release is incomplete.")
    for line in contents.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = release / relative
        require(target.is_file() and sha256(target) == expected,
                f"Frozen presentation checksum mismatch: {relative}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("release_status") ==
            "frozen_post_holdout_synthetic_presentation_training_v2",
            "Frozen presentation release has the wrong status.")
    require(manifest.get("jobs_sha256") == sha256(jobs),
            "Live job matrix differs from the frozen release.")
    for row in read_csv(inventory_path):
        live = repo / row["path"]
        require(live.is_file() and sha256(live) == row["sha256"],
                f"Live source drifted after presentation freeze: {row['path']}")
    return manifest


def freeze(repo: Path, contract_path: Path, jobs: Path, runtime: Path,
           output: Path, archive: Path | None) -> dict[str, Any]:
    require(not output.exists(), f"Presentation release already exists: {output}")
    rows, contract_digest, bundle = validated_rows(
        contract_path, repo, Path("data/synthetic_presentation_response_runs_v2"))
    actual = read_csv(jobs)
    require(actual == [{key: str(value) for key, value in row.items()}
                       for row in rows],
            "Job matrix is not the exact validated 20-job design.")
    require(runtime.is_file(), f"Training runtime evidence not found: {runtime}")
    for relative in SOURCES:
        require((repo / relative).is_file(), f"Required source missing: {relative}")
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
        shutil.copy2(jobs, temporary / "synthetic_presentation_jobs_v2.csv")
        shutil.copy2(repo / json.loads(contract_path.read_text(encoding="utf-8"))[
            "bundle_manifest"], temporary / "synthetic_presentation_bundle_manifest.csv")
        shutil.copy2(runtime, temporary / "training_python_runtime.json")
        manifest = {
            "schema_version": 1,
            "release_status": "frozen_post_holdout_synthetic_presentation_training_v2",
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "job_count": 20, "experiment_count": 2, "seed_count": 10,
            "synthetic_unique_episode_count": 100,
            "synthetic_episode_presentations": 1000,
            "repetition_count": 10,
            "contract_sha256": contract_digest, "jobs_sha256": sha256(jobs),
            "bundle_sha256": bundle["sha256"],
            "bundle_manifest_sha256": bundle["manifest_sha256"],
            "source_100_path_sha256": bundle["source_100_path_sha256"],
            "runtime_sha256": sha256(runtime), "source_count": len(inventory),
            "scientific_note": (
                "This post-holdout experiment separates unique synthetic diversity "
                "from episode presentation/update exposure; it is not confirmatory."),
        }
        (temporary / "synthetic_presentation_release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "READ_ONLY_RELEASE.txt").write_text(
            "Immutable post-holdout synthetic-presentation release. Do not edit.\n",
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
        "publication_pipeline_draft/config/synthetic_presentation_response_v2.json"))
    parser.add_argument("--output-root", type=Path, default=Path(
        "data/synthetic_presentation_response_runs_v2"))
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
                      "seeds": len(contract["seeds"]),
                      "unique_episodes": 100, "episode_presentations": 1000}
        elif args.command == "materialize-jobs":
            require(args.output is not None, "--output is required.")
            rows, digest, bundle = validated_rows(contract_path, repo, args.output_root)
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
        print(f"SYNTHETIC PRESENTATION PROTOCOL FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
