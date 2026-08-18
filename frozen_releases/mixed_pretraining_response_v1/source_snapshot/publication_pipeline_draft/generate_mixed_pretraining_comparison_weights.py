#!/usr/bin/env python3
"""Replay the 20 checkpoints newly authorized for the four-arm comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from publication_pipeline_draft.mixed_pretraining_protocol import (
    JOB_ENV_FIELDS, DoseProtocolError, load_contract, read_csv, require, sha256,
    verify_release,
)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve(); contract, contract_sha = load_contract(args.contract)
    release = verify_release(args.release, repo, args.jobs)
    require(release["contract_sha256"] == contract_sha,
            "Release and mixed contract differ.")
    require(not args.output.exists(), f"Weight output exists: {args.output}")
    require(args.workers >= 1 and args.rscript.is_file() and
            args.policy_python.is_file(), "Inference executable/worker count is invalid.")
    manifest_path = args.audit / "mixed_pretraining_audit_manifest.json"
    table_path = args.audit / "mixed_pretraining_checkpoint_audit.csv"
    require(manifest_path.is_file() and table_path.is_file(),
            "Mixed comparison audit is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_csv(table_path)
    require(manifest.get("status") == "mixed_pretraining_comparison_audit_passed" and
            int(manifest.get("job_count", -1)) == 20 and
            int(manifest.get("experiment_count", -1)) == 2 and
            manifest.get("all_checkpoint_tensors_finite") is True and
            manifest.get("all_checkpoint_metadata_match") is True and
            manifest.get("checkpoint_audit_sha256") == sha256(table_path) and
            len(rows) == 20,
            "Exact passed mixed-pretraining comparison audit is required.")
    for row in rows:
        checkpoint = Path(row["checkpoint"])
        require(checkpoint.is_file() and sha256(checkpoint) ==
                row["checkpoint_sha256"],
                f"Audited checkpoint changed: {row['arm_id']}/{row['seed']}")
    args.output.mkdir(parents=True); logs = args.output / "command_logs"; logs.mkdir()

    def run(row: dict[str, str]) -> dict[str, Any]:
        arm, seed = row["arm_id"], int(row["seed"])
        model = row["checkpoint_model"]
        destination = args.output / "weights" / arm / f"seed_{seed}"
        destination.mkdir(parents=True)
        environment = os.environ.copy()
        environment.update({field: row[field] for field in JOB_ENV_FIELDS})
        environment.update({
            "EVAL_MODEL_DIR": row["model_dir"],
            "EVAL_OUTPUT_DIR": str(destination.resolve()),
            "EVAL_WEIGHTS_ONLY": "true", "EVAL_CHECKPOINT_MODELS": model,
            "EVAL_CHECKPOINT_PREFIX": row["checkpoint_prefix"],
            "EVAL_WINDOW_ID": "locked_oos_v1",
            "EVAL_GATE_AUTHORIZATION": "mixed_pretraining_checkpoint_audit_v1",
            "EVAL_MIXED_AUDIT_MANIFEST": str(manifest_path.resolve()),
            "EVAL_MIXED_CHECKPOINT_AUDIT": str(table_path.resolve()),
            "EVAL_MIXED_CHECKPOINT_SHA256": row["checkpoint_sha256"],
            "POLICY_INFERENCE_SERVER": "rl/policy_inference_server_v2.py",
            "POLICY_PYTHON": str(args.policy_python.resolve()),
            "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C", "TZ": "UTC",
        })
        label = f"{arm}__{seed}__{model}"
        stdout_path = logs / f"{label}.stdout.txt"
        stderr_path = logs / f"{label}.stderr.txt"
        command = [str(args.rscript), "--vanilla", "evaluate_with_config.r",
                   str(args.config), row["model_dir"]]
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            result = subprocess.run(command, cwd=repo, env=environment,
                                    stdout=stdout, stderr=stderr, check=False)
        path = destination / f"weights_rl_{model}_seed_{seed}.csv"
        if result.returncode or not path.is_file():
            tail = stderr_path.read_text(
                encoding="utf-8", errors="replace").splitlines()[-20:]
            raise DoseProtocolError(
                f"Mixed comparison replay failed for {label}: {' | '.join(tail)}")
        return {"experiment_id": arm, "source_experiment_id":
                row["source_experiment_id"], "seed": seed,
                "checkpoint_model": model, "path": str(path.relative_to(repo)),
                "sha256": sha256(path), "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"], "rows": 24}

    inventory: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, row) for row in rows]
        for future in as_completed(futures):
            row = future.result(); inventory.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    inventory.sort(key=lambda row: (row["experiment_id"], row["seed"]))
    inventory_path = args.output / "mixed_pretraining_comparison_weight_manifest.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader(); writer.writerows(inventory)
    result = {
        "schema_version": 1,
        "status": "mixed_pretraining_comparison_weight_replay_complete",
        "evidence_class": "post_holdout_explanatory",
        "confirmatory_claim_permitted": False,
        "policy_count": 20, "experiment_count": 2, "periods_per_policy": 24,
        "new_mixed_full_policy_count": 10,
        "reused_synthetic_pretrained_policy_count": 10,
        "contract_sha256": contract_sha,
        "audit_manifest_sha256": sha256(manifest_path),
        "weight_manifest_sha256": sha256(inventory_path),
    }
    (args.output / "mixed_pretraining_comparison_replay_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/mixed_pretraining_response_v1.json"))
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
    except (DoseProtocolError, OSError, ValueError, KeyError,
            json.JSONDecodeError) as error:
        print(f"MIXED PRETRAINING WEIGHT FAILURE: {error}"); return 1
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
