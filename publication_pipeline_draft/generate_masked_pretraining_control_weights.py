#!/usr/bin/env python3
"""Replay the 20 audited terminal masked pretraining-control checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from publication_pipeline_draft.masked_pretraining_controls_protocol import (
    JOB_ENV_FIELDS, DoseProtocolError, load_contract, read_csv, require, sha256,
    verify_release,
)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve(); contract, contract_sha = load_contract(args.contract)
    release = verify_release(args.release, repo, args.jobs)
    require(release["contract_sha256"] == contract_sha,
            "Release and masked-control contract differ.")
    require(not args.output.exists(), f"Weight output exists: {args.output}")
    require(args.workers >= 1 and args.rscript.is_file() and
            args.policy_python.is_file(), "Inference executable/worker count is invalid.")
    jobs = read_csv(args.jobs)
    audit_manifest_path = args.audit / "synthetic_dose_audit_manifest.json"
    audit_table_path = args.audit / "synthetic_dose_checkpoint_audit.csv"
    require(audit_manifest_path.is_file() and audit_table_path.is_file(),
            "Exact masked-control audit is missing.")
    audit = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    audited_rows = read_csv(audit_table_path)
    require(audit.get("status") == "synthetic_dose_sweep_audit_passed" and
            audit.get("experiment_protocol") ==
            "terminal_masked_pretraining_controls_v1" and
            audit.get("terminal_hpc_experiment") is True and
            int(audit.get("job_count", -1)) == 20 and
            int(audit.get("pretrain_episode_presentations", -1)) == 1000 and
            audit.get("identical_update_budget") is True and
            audit.get("all_checkpoint_tensors_finite") is True and
            audit.get("all_checkpoint_metadata_match") is True,
            "Exact passed terminal masked-control audit is required.")
    job_by_key = {(row["experiment_id"], int(row["seed"])): row for row in jobs}
    audit_by_key = {(row["experiment_id"], int(row["seed"])): row
                    for row in audited_rows}
    require(set(job_by_key) == set(audit_by_key) and len(job_by_key) == 20,
            "Jobs and checkpoint audit differ.")
    require(audit["checkpoint_audit_sha256"] == sha256(audit_table_path),
            "Checkpoint audit hash mismatch.")
    for key, row in audit_by_key.items():
        checkpoint = Path(row["checkpoint"])
        require(checkpoint.is_file() and sha256(checkpoint) ==
                row["checkpoint_sha256"], f"Audited checkpoint changed: {key}")

    args.output.mkdir(parents=True); logs = args.output / "command_logs"; logs.mkdir()

    def run(key: tuple[str, int]) -> dict[str, Any]:
        experiment, seed = key; job, audited = job_by_key[key], audit_by_key[key]
        destination = args.output / "weights" / experiment / f"seed_{seed}"
        destination.mkdir(parents=True)
        model_dir = (repo / job["output_dir"]).resolve()
        environment = os.environ.copy()
        environment.update({field: job[field] for field in JOB_ENV_FIELDS})
        environment.update({
            "EVAL_MODEL_DIR": str(model_dir), "EVAL_OUTPUT_DIR": str(destination.resolve()),
            "EVAL_WEIGHTS_ONLY": "true", "EVAL_CHECKPOINT_MODELS": "full",
            "EVAL_CHECKPOINT_PREFIX": job["CHECKPOINT_PREFIX"],
            "EVAL_WINDOW_ID": "locked_oos_v1",
            "EVAL_GATE_AUTHORIZATION": "synthetic_dose_checkpoint_audit_v1",
            "EVAL_DOSE_AUDIT_MANIFEST": str(audit_manifest_path.resolve()),
            "EVAL_DOSE_CHECKPOINT_AUDIT": str(audit_table_path.resolve()),
            "EVAL_DOSE_CHECKPOINT_SHA256": audited["checkpoint_sha256"],
            "POLICY_INFERENCE_SERVER": "rl/policy_inference_server_v2.py",
            "POLICY_PYTHON": str(args.policy_python.resolve()),
            "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C", "TZ": "UTC",
        })
        label = f"{experiment}__{seed}"
        stdout_path, stderr_path = (logs / f"{label}.stdout.txt",
                                    logs / f"{label}.stderr.txt")
        command = [str(args.rscript), "--vanilla", "evaluate_with_config.r",
                   str(args.config), str(model_dir)]
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            result = subprocess.run(command, cwd=repo, env=environment,
                                    stdout=stdout, stderr=stderr, check=False)
        path = destination / f"weights_rl_full_seed_{seed}.csv"
        if result.returncode or not path.is_file():
            tail = stderr_path.read_text(
                encoding="utf-8", errors="replace").splitlines()[-20:]
            raise DoseProtocolError(
                f"Masked-control replay failed for {label}: {' | '.join(tail)}")
        return {"experiment_id": experiment, "seed": seed,
                "path": str(path.relative_to(repo)), "sha256": sha256(path),
                "checkpoint": audited["checkpoint"],
                "checkpoint_sha256": audited["checkpoint_sha256"], "rows": 24}

    inventory: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, key) for key in sorted(job_by_key)]
        for future in as_completed(futures):
            row = future.result(); inventory.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    inventory.sort(key=lambda row: (row["experiment_id"], row["seed"]))
    inventory_path = args.output / "masked_pretraining_control_weight_manifest.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader(); writer.writerows(inventory)
    manifest = {
        "schema_version": 1,
        "status": "terminal_masked_pretraining_control_replay_complete",
        "evidence_class": "post_holdout_explanatory",
        "confirmatory_claim_permitted": False, "terminal_hpc_experiment": True,
        "policy_count": 20, "experiment_count": 2, "periods_per_policy": 24,
        "contract_sha256": contract_sha,
        "release_contents_sha256": sha256(args.release / "CONTENTS.sha256"),
        "audit_manifest_sha256": sha256(audit_manifest_path),
        "weight_manifest_sha256": sha256(inventory_path),
    }
    (args.output / "masked_pretraining_control_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/masked_pretraining_controls_v1.json"))
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--policy-python", required=True, type=Path)
    parser.add_argument("--rscript", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); args.repo_root = args.repo_root.resolve()
    args.contract = (args.repo_root / args.contract).resolve()
    args.config = (args.repo_root / args.config).resolve()
    for name in ("release", "jobs", "audit", "policy_python", "rscript", "output"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        result = generate(args)
    except (DoseProtocolError, OSError, ValueError, KeyError) as error:
        print(f"MASKED PRETRAINING CONTROL REPLAY FAILURE: {error}"); return 1
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
