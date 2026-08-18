#!/usr/bin/env python3
"""Replay 15 audited focused-window checkpoints into immutable weight logs."""

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

from publication_pipeline_draft.focused_window_training_protocol import ENV_FIELDS
from publication_pipeline_draft.run_focused_window_sweep import verify_contract


class FocusedWeightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FocusedWeightError(message)


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
    contract, jobs = verify_contract(args.contract.resolve())
    require(not args.output.exists(), f"Output already exists: {args.output}")
    require(args.workers >= 1 and args.rscript.is_file() and
            args.policy_python.is_file(),
            "Inference worker count or executable is invalid.")
    audit_manifest_path = args.audit / "focused_sweep_audit_manifest.json"
    audit_table_path = args.audit / "focused_checkpoint_audit.csv"
    audit = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    audit_rows = read_csv(audit_table_path)
    require(audit.get("status") == "focused_window_sweep_audit_passed" and
            int(audit.get("job_count", -1)) == 15 and
            audit.get("all_checkpoint_tensors_finite") is True and
            audit.get("all_checkpoint_metadata_match") is True and
            audit.get("all_behavior_gate_enforcement_valid") is True,
            "Exact passed focused checkpoint audit is required.")
    job_by_key = {(row["experiment_id"], int(row["seed"])): row for row in jobs}
    audit_by_key = {(row["experiment_id"], int(row["seed"])): row
                    for row in audit_rows}
    require(set(job_by_key) == set(audit_by_key) and len(job_by_key) == 15,
            "Focused jobs and checkpoint audit differ.")
    require(audit.get("checkpoint_audit_sha256") == sha256(audit_table_path),
            "Focused checkpoint audit hash mismatch.")
    for key, row in audit_by_key.items():
        checkpoint = Path(row["checkpoint"])
        require(checkpoint.is_file() and
                sha256(checkpoint) == row["checkpoint_sha256"],
                f"Audited checkpoint changed: {key}")

    args.output.mkdir(parents=True)
    logs = args.output / "command_logs"
    logs.mkdir()

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
            "EVAL_WEIGHTS_ONLY": "true", "EVAL_CHECKPOINT_MODELS": "full",
            "EVAL_CHECKPOINT_PREFIX": job["CHECKPOINT_PREFIX"],
            "EVAL_WINDOW_ID": contract["window_id"],
            "EVAL_GATE_AUTHORIZATION": "focused_checkpoint_audit_v1",
            "EVAL_FOCUSED_AUDIT_MANIFEST": str(audit_manifest_path.resolve()),
            "EVAL_FOCUSED_CHECKPOINT_AUDIT": str(audit_table_path.resolve()),
            "EVAL_FOCUSED_CHECKPOINT_SHA256": audited["checkpoint_sha256"],
            "POLICY_INFERENCE_SERVER": "rl/policy_inference_server_v2.py",
            "POLICY_PYTHON": str(args.policy_python.resolve()),
            "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C", "TZ": "UTC",
        })
        label = f"{experiment}__{seed}"
        stdout_path = logs / f"{label}.stdout.txt"
        stderr_path = logs / f"{label}.stderr.txt"
        command = [str(args.rscript.resolve()), "--vanilla",
                   "evaluate_with_config.r", str(args.config.resolve()),
                   str(model_dir)]
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            result = subprocess.run(command, cwd=repo, env=environment,
                                    stdout=stdout, stderr=stderr, check=False)
        path = destination / f"weights_rl_full_seed_{seed}.csv"
        if result.returncode or not path.is_file():
            tail = stderr_path.read_text(
                encoding="utf-8", errors="replace").splitlines()[-20:]
            raise FocusedWeightError(
                f"Focused replay failed for {label}: {' | '.join(tail)}")
        return {"window_id": contract["window_id"],
                "experiment_id": experiment, "seed": seed,
                "weight_file": str(path.resolve()), "sha256": sha256(path),
                "checkpoint": str(Path(audited["checkpoint"]).resolve()),
                "checkpoint_sha256": audited["checkpoint_sha256"],
                "rows": 24}

    inventory: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, key) for key in sorted(job_by_key)]
        for future in as_completed(futures):
            row = future.result()
            inventory.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    inventory.sort(key=lambda row: (row["experiment_id"], row["seed"]))
    path = args.output / "focused_policy_weight_inventory.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    manifest = {"schema_version": 1,
                "status": "focused_window_policy_weight_replay_complete",
                "window_id": contract["window_id"], "policy_count": 15,
                "experiment_count": 3, "seed_count": 5,
                "inventory_sha256": sha256(path),
                "checkpoint_audit_sha256": sha256(audit_table_path),
                "confirmatory_claim_permitted": False}
    (args.output / "focused_policy_weight_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--policy-python", required=True, type=Path)
    parser.add_argument("--rscript", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.config = (args.repo_root / args.config).resolve()
    args.audit = args.audit.resolve()
    try:
        result = generate(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            FocusedWeightError) as error:
        print(f"FOCUSED POLICY WEIGHT FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
