#!/usr/bin/env python3
"""Run a frozen external-window RL comparison on matched seeds and GPUs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from publication_pipeline_draft.extension_release import (
    ExtensionReleaseError, verify_extension_release,
)


class WindowSweepError(RuntimeError):
    pass


ENV_FIELDS = (
    "RETURNS_DATA_FILE", "RETURNS_DATA_KIND", "RETURNS_DATA_MANIFEST",
    "REF_COL", "VINE_TRUNCATION_LEVEL", "SYNTHETIC_RETURNS_FILE",
    "TRAINING_MARGINALS_FILE", "NN_VINE_MODEL_DIR", "FINETUNE_RETURNS_FILE",
    "RL_ALGORITHM", "POLICY_ENCODER", "VINE_OBSERVATION_MODE",
    "VINE_FEATURE_MODE", "CVAR_OBSERVATION_MODE", "CVAR_REWARD_MODE",
    "PRETRAIN_DATA_MODE", "RUN_FINETUNE", "CHECKPOINT_PREFIX",
    "LR_ACTOR", "LR_CRITIC", "ENTROPY_COEF",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_contract(root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest_path = root / "window_training_manifest.json"
    jobs_path = root / "window_rl_jobs.csv"
    contents = root / "CONTENTS.sha256"
    if not all(path.is_file() for path in (manifest_path, jobs_path, contents)):
        raise WindowSweepError("Frozen window training contract is incomplete.")
    for line in contents.read_text(encoding="ascii").splitlines():
        if line.strip():
            expected, relative = line.split("  ", 1)
            target = root / relative
            if not target.is_file() or sha256(target) != expected:
                raise WindowSweepError(f"Contract checksum mismatch: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("release_status") != \
            "frozen_development_window_training_contract":
        raise WindowSweepError("Window contract status is invalid.")
    if manifest.get("jobs_sha256") != sha256(jobs_path):
        raise WindowSweepError("Window job matrix hash mismatch.")
    with jobs_path.open(newline="", encoding="utf-8") as stream:
        jobs = list(csv.DictReader(stream))
    required = {"window_id", "panel_id", "algorithm", "seed", "output_dir",
                *ENV_FIELDS}
    if not jobs or not required <= set(jobs[0]):
        raise WindowSweepError("Window job matrix is empty or incomplete.")
    if len(jobs) != 50 or len({(row["algorithm"], row["seed"]) for row in jobs}) != 50:
        raise WindowSweepError("Window comparison must contain 5 algorithms x 10 seeds.")
    if {row["algorithm"] for row in jobs} != {"td3", "ddpg", "sac", "ppo", "a2c"}:
        raise WindowSweepError("Window RL algorithm family is incomplete.")
    return manifest, jobs


def run_job(row: dict[str, str], gpu: int, args: argparse.Namespace,
            bundle_metadata: dict[str, Any], cores: int) -> dict[str, Any]:
    repo = args.repo_root
    output = (repo / row["output_dir"]).resolve()
    if output.exists():
        raise WindowSweepError(f"Immutable policy output exists: {output}")
    label = f"{row['algorithm']}__{row['seed']}"
    environment = os.environ.copy()
    environment.update({field: row[field] for field in ENV_FIELDS})
    environment.update({
        "TRAIN_SEED": row["seed"], "TRAIN_OUTPUT_DIR": str(output),
        "TRAIN_DEVICE": "cuda", "CUDA_VISIBLE_DEVICES": str(gpu),
        "VINE_SIM_CORES": str(cores), "OMP_NUM_THREADS": str(cores),
        "MKL_NUM_THREADS": str(cores), "RETICULATE_PYTHON": str(args.train_python),
        "POLICY_PYTHON": str(args.train_python),
        "PRETRAIN_EPISODES": str(bundle_metadata["pretrain_episodes"]),
        "FINETUNE_EPISODES": str(bundle_metadata["finetune_episodes"]),
        "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C", "TZ": "UTC",
    })
    start = time.monotonic()
    train_stdout = args.log_root / f"{label}.train.stdout.txt"
    train_stderr = args.log_root / f"{label}.train.stderr.txt"
    train_command = [str(args.rscript), "--vanilla", "run_with_config.r",
                     str(args.config)]
    with train_stdout.open("wb") as stdout, train_stderr.open("wb") as stderr:
        train = subprocess.run(train_command, cwd=repo, env=environment,
                               stdout=stdout, stderr=stderr, check=False)
    sanity_exit = None
    if train.returncode == 0:
        sanity_stdout = args.log_root / f"{label}.sanity.stdout.txt"
        sanity_stderr = args.log_root / f"{label}.sanity.stderr.txt"
        sanity_command = [str(args.rscript), "--vanilla",
                          "rl/training_sanity_check.r", str(args.config)]
        with sanity_stdout.open("wb") as stdout, sanity_stderr.open("wb") as stderr:
            sanity = subprocess.run(sanity_command, cwd=repo, env=environment,
                                    stdout=stdout, stderr=stderr, check=False)
        sanity_exit = sanity.returncode
    checkpoint = output / f"{row['CHECKPOINT_PREFIX']}_full.pt"
    report = output / "sanity_no_holdout" / "sanity_report.json"
    report_pass = False
    if report.is_file():
        value = json.loads(report.read_text(encoding="utf-8"))
        report_pass = bool(value.get("overall_pass")) and bool(
            value.get("publication_behavior_pass"))
    passed = (train.returncode == 0 and sanity_exit == 0 and checkpoint.is_file()
              and report_pass)
    return {
        "window_id": row["window_id"], "panel_id": row["panel_id"],
        "algorithm": row["algorithm"], "seed": int(row["seed"]), "gpu": gpu,
        "train_exit_code": train.returncode, "sanity_exit_code": sanity_exit,
        "checkpoint_exists": checkpoint.is_file(), "sanity_pass": report_pass,
        "passed": passed, "duration_seconds": time.monotonic() - start,
        "output_dir": row["output_dir"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--train-python", required=True, type=Path)
    parser.add_argument("--rscript", required=True, type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--cpu-cores", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    args = parser.parse_args()
    try:
        args.repo_root = args.repo_root.resolve()
        args.config = (args.repo_root / args.config).resolve()
        release = verify_extension_release(args.release, args.repo_root)
        manifest, jobs = verify_contract(args.contract.resolve())
        if manifest.get("program_sha256") != release.get("program_sha256"):
            raise WindowSweepError("Window contract and extension release differ.")
        if not args.train_python.is_file() or not args.rscript.is_file():
            raise WindowSweepError("Training Python/Rscript executable is missing.")
        if args.log_root.exists() or args.status.exists():
            raise WindowSweepError("Immutable status/log output already exists.")
        bundle_path = args.repo_root / jobs[0]["SYNTHETIC_RETURNS_FILE"]
        bundle_manifest_path = bundle_path.parent / "synthetic_bundle_manifest.json"
        if not bundle_path.is_file() or not bundle_manifest_path.is_file():
            raise WindowSweepError("Attested window synthetic bundle is missing.")
        bundle = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
        if bundle.get("bundle_sha256") != sha256(bundle_path) or \
                bundle.get("window_id") != manifest["window_id"] or \
                bundle.get("diagnostics_passed") is not True:
            raise WindowSweepError("Window synthetic bundle attestation failed.")
        gpus = [int(item) for item in args.gpus.split(",")]
        if not gpus or len(gpus) != len(set(gpus)) or min(gpus) < 0:
            raise WindowSweepError("GPU identifiers are invalid.")
        for row in jobs:
            if (args.repo_root / row["output_dir"]).exists():
                raise WindowSweepError(f"Job output already exists: {row['output_dir']}")
        args.log_root.mkdir(parents=True)
        args.status.parent.mkdir(parents=True, exist_ok=True)
        cores = max(1, args.cpu_cores // len(gpus))
        locks = {gpu: threading.Lock() for gpu in gpus}

        def assigned(item: tuple[int, dict[str, str]]) -> dict[str, Any]:
            index, row = item; gpu = gpus[index % len(gpus)]
            with locks[gpu]:
                return run_job(row, gpu, args, bundle, cores)

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(assigned, item) for item in enumerate(jobs)]
            for future in as_completed(futures):
                result = future.result(); results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
        results.sort(key=lambda row: (row["algorithm"], row["seed"]))
        with args.status.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader(); writer.writerows(results)
        passed = sum(bool(row["passed"]) for row in results)
        summary = {"status": "complete" if passed == len(results) else "failed",
                   "jobs": len(results), "passed": passed,
                   "confirmatory_claim_permitted": False}
        print(json.dumps(summary, indent=2))
        return 0 if passed == len(results) else 1
    except (OSError, ValueError, json.JSONDecodeError, WindowSweepError,
            ExtensionReleaseError) as error:
        print(f"WINDOW RL SWEEP FAILURE: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
