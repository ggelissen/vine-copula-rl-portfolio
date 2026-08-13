#!/usr/bin/env python3
"""Replay all 130 audited causal checkpoints into immutable target-weight logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from publication_pipeline_draft.causal_ablation_protocol import ENV_FIELDS
from publication_pipeline_draft.causal_analysis_contract import load_contract, require
from publication_pipeline_draft.freeze_causal_analysis_plan import (
    CausalAnalysisFreezeError,
    verify_causal_analysis_release,
)


class CausalWeightError(CausalAnalysisFreezeError):
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


def generate(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    analysis_release = verify_causal_analysis_release(args.analysis_release, repo)
    contract = load_contract(args.contract.resolve())
    require(analysis_release["analysis_contract_sha256"] == contract.sha256,
            "Analysis release and live causal contract differ.")
    require(not args.output.exists(), f"Weight output already exists: {args.output}")
    require(args.workers >= 1, "workers must be positive.")
    require(args.rscript.is_file(), f"Rscript not found: {args.rscript}")
    require(args.policy_python.is_file(), f"Policy Python not found: {args.policy_python}")

    jobs = read_csv(args.jobs)
    audit = json.loads((args.audit / "causal_sweep_audit_manifest.json").read_text(
        encoding="utf-8"))
    audit_rows = read_csv(args.audit / "checkpoint_audit.csv")
    require(audit.get("status") == "causal_sweep_audit_passed" and
            int(audit.get("job_count", -1)) == 130,
            "The exact passed 130-checkpoint audit is required.")
    require(audit.get("all_checkpoint_tensors_finite") is True and
            audit.get("all_behavior_gate_enforcement_valid") is True and
            audit.get("all_checkpoint_metadata_match") is True and
            audit.get("mixed_revision_carry_forward") is True,
            "The checkpoint audit lacks required finite/metadata/gate evidence.")
    require(int(audit.get("v2_carried_count", -1)) == 70 and
            int(audit.get("v3_carried_count", -1)) == 31 and
            int(audit.get("v4_retry_count", -1)) == 29,
            "The checkpoint audit is not the disclosed 70/31/29 evidence.")
    expected = {(experiment, seed) for experiment in contract.experiment_ids
                for seed in contract.raw["expected_seeds"]}
    job_by_key = {(row["experiment_id"], int(row["seed"])): row for row in jobs}
    audit_by_key = {(row["experiment_id"], int(row["seed"])): row
                    for row in audit_rows}
    require(set(job_by_key) == set(audit_by_key) == expected,
            "Jobs and checkpoint audit are not the exact 13-by-10 design.")
    require(sha256(args.jobs) == audit["jobs_sha256"],
            "Job matrix changed after the checkpoint audit.")
    for key, row in audit_by_key.items():
        checkpoint = repo / row["checkpoint"]
        require(checkpoint.is_file() and sha256(checkpoint) == row["sha256"],
                f"Audited checkpoint changed: {key}")

    args.output.mkdir(parents=True)
    log_root = args.output / "command_logs"
    log_root.mkdir()

    def run(key: tuple[str, int]) -> dict[str, Any]:
        experiment, seed = key
        job, audited = job_by_key[key], audit_by_key[key]
        destination = args.output / "weights" / experiment / f"seed_{seed}"
        destination.mkdir(parents=True)
        model_dir = (repo / job["output_dir"]).resolve()
        environment = os.environ.copy()
        environment.update({field: job[field] for field in ENV_FIELDS})
        environment.update({
            "EVAL_MODEL_DIR": str(model_dir),
            "EVAL_OUTPUT_DIR": str(destination.resolve()),
            "EVAL_WEIGHTS_ONLY": "true",
            "EVAL_CHECKPOINT_MODELS": "full",
            "EVAL_CHECKPOINT_PREFIX": job["CHECKPOINT_PREFIX"],
            "EVAL_WINDOW_ID": contract.raw["sample"]["window_id"],
            # This is not a general gate bypass.  evaluate_with_config.r reads
            # the frozen audit itself and authorizes only this exact checkpoint
            # hash for weights-only replay.
            "EVAL_GATE_AUTHORIZATION": "causal_checkpoint_audit_v1",
            "EVAL_CAUSAL_AUDIT_MANIFEST": str(
                (args.audit / "causal_sweep_audit_manifest.json").resolve()),
            "EVAL_CAUSAL_CHECKPOINT_AUDIT": str(
                (args.audit / "checkpoint_audit.csv").resolve()),
            "EVAL_CAUSAL_CHECKPOINT_SHA256": audited["sha256"],
            "POLICY_INFERENCE_SERVER": "rl/policy_inference_server_v2.py",
            "POLICY_PYTHON": str(args.policy_python.resolve()),
            "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C", "TZ": "UTC",
        })
        label = f"{experiment}__{seed}"
        stdout_path = log_root / f"{label}.stdout.txt"
        stderr_path = log_root / f"{label}.stderr.txt"
        command = [str(args.rscript.resolve()), "--vanilla",
                   "evaluate_with_config.r", str(args.config.resolve()), str(model_dir)]
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            result = subprocess.run(command, cwd=repo, env=environment,
                                    stdout=stdout, stderr=stderr, check=False)
        weight_path = destination / f"weights_rl_full_seed_{seed}.csv"
        if result.returncode != 0 or not weight_path.is_file():
            diagnostic = stderr_path.read_text(
                encoding="utf-8", errors="replace").strip().splitlines()
            tail = "\n".join(diagnostic[-20:]) or "<empty stderr>"
            raise CausalWeightError(
                f"Causal policy replay failed for {label}; inspect {stderr_path}.\n"
                f"Last stderr lines:\n{tail}")
        return {
            "experiment_id": experiment, "seed": seed,
            "path": str(weight_path.relative_to(repo)), "sha256": sha256(weight_path),
            "checkpoint": audited["checkpoint"],
            "checkpoint_sha256": audited["sha256"],
            "evaluation_exit_code": result.returncode,
        }

    inventory: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, key): key for key in sorted(expected)}
        for future in as_completed(futures):
            row = future.result()
            inventory.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    inventory.sort(key=lambda row: (row["experiment_id"], row["seed"]))
    inventory_path = args.output / "causal_policy_weight_manifest.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader(); writer.writerows(inventory)
    manifest = {
        "schema_version": 1,
        "status": "causal_policy_weight_replay_complete",
        "analysis_contract_sha256": contract.sha256,
        "analysis_release_contents_sha256": analysis_release["release_contents_sha256"],
        "policy_count": len(inventory), "experiment_count": 13,
        "periods_per_policy": 24,
        "weight_manifest_sha256": sha256(inventory_path),
        "audit_manifest_sha256": sha256(
            args.audit / "causal_sweep_audit_manifest.json"),
        "audited_jobs_sha256": audit["jobs_sha256"],
        "evidence_class": "post_holdout_explanatory",
        "confirmatory_claim_permitted": False,
        "gate_authorization": "causal_checkpoint_audit_v1",
        "operational_revision": "replay_v2_after_legacy_local_sanity_mismatch",
        "prior_failed_replay_exported_policy_count": 0,
    }
    (args.output / "causal_policy_weight_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v1.json"))
    parser.add_argument("--analysis-release", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--policy-python", required=True, type=Path)
    parser.add_argument("--rscript", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.contract = (args.repo_root / args.contract).resolve()
    args.config = (args.repo_root / args.config).resolve()
    args.analysis_release = args.analysis_release.resolve()
    args.jobs = args.jobs.resolve(); args.audit = args.audit.resolve()
    args.rscript = args.rscript.resolve(); args.policy_python = args.policy_python.resolve()
    args.output = args.output.resolve()
    try:
        result = generate(args)
    except (CausalAnalysisFreezeError, OSError, ValueError, KeyError,
            json.JSONDecodeError) as error:
        print(f"CAUSAL POLICY WEIGHT FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
