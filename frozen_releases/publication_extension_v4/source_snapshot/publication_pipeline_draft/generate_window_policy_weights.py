#!/usr/bin/env python3
"""Generate immutable OOS weight logs for every passed window policy."""

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

from publication_pipeline_draft.run_window_rl_sweep import ENV_FIELDS, verify_contract
from publication_pipeline_draft.extension_release import (
    ExtensionReleaseError, verify_extension_release,
)


class WeightGenerationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--sweep-status", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--policy-python", required=True, type=Path)
    parser.add_argument("--rscript", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        repo = args.repo_root.resolve()
        release = verify_extension_release(args.release, repo)
        manifest, jobs = verify_contract(args.contract.resolve())
        if manifest.get("program_sha256") != release.get("program_sha256"):
            raise WeightGenerationError("Window contract and extension release differ.")
        if args.output.exists() or args.workers < 1:
            raise WeightGenerationError("Output exists or worker count is invalid.")
        if not args.policy_python.is_file() or not args.rscript.is_file():
            raise WeightGenerationError("Policy Python/Rscript executable is missing.")
        with args.sweep_status.open(newline="", encoding="utf-8") as stream:
            statuses = list(csv.DictReader(stream))
        status_by_key = {(row["algorithm"], row["seed"]): row for row in statuses}
        keys = {(row["algorithm"], row["seed"]) for row in jobs}
        if set(status_by_key) != keys or any(
                row.get("passed", "").lower() not in {"true", "1"}
                for row in statuses):
            raise WeightGenerationError("Exact 50-job passed sweep status is required.")
        args.output.mkdir(parents=True)

        def run(row: dict[str, str]) -> dict[str, Any]:
            label = f"{row['algorithm']}__{row['seed']}"
            destination = args.output / label
            destination.mkdir()
            model_dir = (repo / row["output_dir"]).resolve()
            environment = os.environ.copy()
            environment.update({field: row[field] for field in ENV_FIELDS})
            environment.update({
                "EVAL_MODEL_DIR": str(model_dir),
                "EVAL_OUTPUT_DIR": str(destination.resolve()),
                "EVAL_WEIGHTS_ONLY": "true", "EVAL_CHECKPOINT_MODELS": "full",
                "EVAL_CHECKPOINT_PREFIX": row["CHECKPOINT_PREFIX"],
                "EVAL_WINDOW_ID": row["window_id"],
                "POLICY_INFERENCE_SERVER": "rl/policy_inference_server_v2.py",
                "POLICY_PYTHON": str(args.policy_python),
                "RETICULATE_PYTHON": str(args.policy_python),
                "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C", "TZ": "UTC",
            })
            command = [str(args.rscript), "--vanilla", "evaluate_with_config.r",
                       str((repo / args.config).resolve()), str(model_dir)]
            stdout_path = destination / "evaluation.stdout.txt"
            stderr_path = destination / "evaluation.stderr.txt"
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                result = subprocess.run(command, cwd=repo, env=environment,
                                        stdout=stdout, stderr=stderr, check=False)
            expected = destination / f"weights_rl_full_seed_{row['seed']}.csv"
            if result.returncode or not expected.is_file():
                raise WeightGenerationError(
                    f"Policy weight generation failed for {label}; inspect {destination}.")
            return {"window_id": row["window_id"], "algorithm": row["algorithm"],
                    "seed": int(row["seed"]), "weight_file": str(expected),
                    "sha256": sha256(expected), "rows": 24}

        inventory: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run, row) for row in jobs]
            for future in as_completed(futures):
                result = future.result(); inventory.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
        inventory.sort(key=lambda row: (row["algorithm"], row["seed"]))
        inventory_path = args.output / "policy_weight_inventory.csv"
        with inventory_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
            writer.writeheader(); writer.writerows(inventory)
        result_manifest = {
            "schema_version": 1, "status": "complete_window_policy_weights",
            "window_id": manifest["window_id"], "policy_count": len(inventory),
            "algorithm_count": 5, "seed_count": 10,
            "inventory_sha256": sha256(inventory_path),
            "confirmatory_claim_permitted": False,
        }
        (args.output / "policy_weight_manifest.json").write_text(
            json.dumps(result_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps(result_manifest, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError,
            WeightGenerationError, ExtensionReleaseError) as error:
        print(f"WINDOW POLICY WEIGHT FAILURE: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
