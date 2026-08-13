#!/usr/bin/env python3
"""Run a preregistered causal job matrix over one or more GPUs.

This runner accesses training data only. It refuses existing output paths,
records every command/environment, never substitutes failed seeds, and emits a
single status table suitable for the later freeze gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from publication_pipeline_draft.behavior_gate_protocol import (
    BehaviorGateProtocolError,
    validate_report_only_trainer,
)


ENV_FIELDS = (
    "RL_ALGORITHM", "POLICY_ENCODER", "VINE_FEATURE_MODE",
    "CVAR_OBSERVATION_MODE", "CVAR_REWARD_MODE", "PRETRAIN_DATA_MODE",
    "RUN_FINETUNE", "SYNTHETIC_RETURNS_FILE", "CHECKPOINT_PREFIX",
    "LR_ACTOR", "LR_CRITIC", "ENTROPY_COEF",
    "PRETRAIN_BEHAVIOR_GATE_MODE",
)


class SweepError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_release(release: Path, repo_root: Path, jobs: Path) -> None:
    manifest_path = release / "publication_extension_release_manifest.json"
    inventory_path = release / "source_inventory.csv"
    checksums = release / "CONTENTS.sha256"
    if not all(path.is_file() for path in (manifest_path, inventory_path, checksums)):
        raise SweepError("Frozen extension release is incomplete.")
    for line in checksums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = release / relative
        if not target.is_file() or sha256(target) != expected:
            raise SweepError(f"Frozen release checksum mismatch: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("release_status") != "frozen_pre_external_test_publication_extension":
        raise SweepError("Release status does not authorize the extension sweep.")
    if manifest.get("causal_jobs_sha256") != sha256(jobs):
        raise SweepError("Selected causal job matrix differs from the frozen release.")
    job_rows = read_inventory(jobs)
    if any(row.get("PRETRAIN_BEHAVIOR_GATE_MODE") == "report_only"
           for row in job_rows):
        try:
            validate_report_only_trainer(
                release / "source_snapshot/rl/train_rl.r")
        except BehaviorGateProtocolError as error:
            raise SweepError(str(error)) from error
        if manifest.get("report_only_gate_wiring_valid") is not True:
            raise SweepError(
                "Frozen release does not attest report-only gate wiring.")
    for row in read_inventory(inventory_path):
        live = repo_root / row["path"]
        if not live.is_file() or sha256(live) != row["sha256"]:
            raise SweepError(f"Live source drifted after freeze: {row['path']}")


def load_jobs(path: Path, experiments: set[str] | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if experiments:
        rows = [row for row in rows if row["experiment_id"] in experiments]
    if not rows:
        raise SweepError("No jobs selected.")
    required = {"experiment_id", "seed", "output_dir", *ENV_FIELDS}
    if not required <= set(rows[0]):
        raise SweepError(f"Job matrix is missing: {sorted(required - set(rows[0]))}")
    keys = [(row["experiment_id"], row["seed"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise SweepError("Selected matrix contains duplicate experiment/seed jobs.")
    return rows


def select_failed_retry_jobs(
        jobs: list[dict[str, str]], status_path: Path,
        expected_count: int | None) -> list[dict[str, str]]:
    """Select the exact failed keys from an immutable prior status table."""
    statuses = read_inventory(status_path)
    if not statuses:
        raise SweepError("Prior retry status is empty.")
    required = {"experiment_id", "seed", "passed"}
    if not required <= set(statuses[0]):
        raise SweepError(
            f"Prior retry status is missing: {sorted(required - set(statuses[0]))}")
    status_keys = [(row["experiment_id"], row["seed"]) for row in statuses]
    if len(status_keys) != len(set(status_keys)):
        raise SweepError("Prior retry status contains duplicate keys.")
    failed = {(row["experiment_id"], row["seed"]) for row in statuses
              if row["passed"].strip().lower() not in {"1", "true", "yes"}}
    if expected_count is not None and len(failed) != expected_count:
        raise SweepError(
            f"Expected {expected_count} failed retry keys; found {len(failed)}.")
    by_key = {(row["experiment_id"], row["seed"]): row for row in jobs}
    missing = failed - set(by_key)
    if missing:
        raise SweepError(f"Prior failed keys are absent from current jobs: {sorted(missing)}")
    selected = [by_key[key] for key in sorted(failed)]
    if not selected:
        raise SweepError("Prior status contains no failed jobs to retry.")
    return selected


def run_job(row: dict[str, str], gpu: int, args: argparse.Namespace,
            log_root: Path, cpu_cores: int) -> dict[str, Any]:
    output = (args.repo_root / row["output_dir"]).resolve()
    if output.exists():
        raise SweepError(f"Immutable job output already exists: {output}")
    bundle = (args.repo_root / row["SYNTHETIC_RETURNS_FILE"]).resolve()
    if not bundle.is_file():
        raise SweepError(f"Training bundle not found: {bundle}")
    environment = os.environ.copy()
    environment.update({field: row[field] for field in ENV_FIELDS})
    environment.update({
        "TRAIN_SEED": row["seed"],
        "TRAIN_OUTPUT_DIR": str(output),
        "TRAIN_DEVICE": "cuda",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "VINE_SIM_CORES": str(cpu_cores),
        "OMP_NUM_THREADS": str(cpu_cores),
        "MKL_NUM_THREADS": str(cpu_cores),
        "RETICULATE_PYTHON": str(args.train_python),
        "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C", "TZ": "UTC",
    })
    label = f"{row['experiment_id']}__{row['seed']}"
    stdout_path = log_root / f"{label}.stdout.txt"
    stderr_path = log_root / f"{label}.stderr.txt"
    command = [str(args.rscript), "--vanilla", "run_with_config.r", str(args.config)]
    start = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        result = subprocess.run(command, cwd=args.repo_root, env=environment,
                                stdout=stdout, stderr=stderr, check=False)
    duration = time.monotonic() - start
    checkpoint = output / f"{row['CHECKPOINT_PREFIX']}_full.pt"
    gate = output / "pretraining_behavior_gate.csv"
    return {
        "experiment_id": row["experiment_id"], "seed": int(row["seed"]),
        "gpu": gpu, "exit_code": result.returncode,
        "duration_seconds": duration, "output_dir": row["output_dir"],
        "checkpoint_exists": checkpoint.is_file(), "gate_exists": gate.is_file(),
        "passed": result.returncode == 0 and checkpoint.is_file() and gate.is_file(),
        "stdout": str(stdout_path), "stderr": str(stderr_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--train-python", required=True, type=Path)
    parser.add_argument("--rscript", required=True, type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--cpu-cores", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--experiments", default="")
    parser.add_argument("--retry-failures-from", type=Path)
    parser.add_argument("--expected-selected-jobs", type=int)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.config = (args.repo_root / args.config).resolve()
    experiments = {value.strip() for value in args.experiments.split(",") if value.strip()}
    try:
        if experiments and args.retry_failures_from is not None:
            raise SweepError(
                "Use either --experiments or --retry-failures-from, not both.")
        jobs = load_jobs(args.jobs, None)
        validate_release(args.release.resolve(), args.repo_root, args.jobs.resolve())
        if experiments:
            jobs = [row for row in jobs if row["experiment_id"] in experiments]
            if not jobs:
                raise SweepError("No jobs matched --experiments.")
        if args.retry_failures_from is not None:
            jobs = select_failed_retry_jobs(
                jobs, args.retry_failures_from.resolve(),
                args.expected_selected_jobs)
        elif (args.expected_selected_jobs is not None and
              len(jobs) != args.expected_selected_jobs):
            raise SweepError(
                f"Expected {args.expected_selected_jobs} selected jobs; "
                f"found {len(jobs)}.")
        gpus = [int(value) for value in args.gpus.split(",")]
        if len(gpus) != len(set(gpus)) or any(value < 0 for value in gpus):
            raise SweepError("GPU identifiers must be distinct non-negative integers.")
        if args.status.exists() or args.log_root.exists():
            raise SweepError("Sweep status/log output already exists.")
        if not args.train_python.is_file() or not args.rscript.is_file():
            raise SweepError("Training Python or Rscript executable was not found.")
        for row in jobs:
            if (args.repo_root / row["output_dir"]).exists():
                raise SweepError(f"Job output already exists: {row['output_dir']}")
        args.log_root.mkdir(parents=True)
        args.status.parent.mkdir(parents=True, exist_ok=True)
        cores_per_job = max(1, args.cpu_cores // len(gpus))
        locks = {gpu: threading.Lock() for gpu in gpus}

        def assigned(index_and_row: tuple[int, dict[str, str]]) -> dict[str, Any]:
            index, row = index_and_row
            gpu = gpus[index % len(gpus)]
            with locks[gpu]:
                return run_job(row, gpu, args, args.log_root, cores_per_job)

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(assigned, value) for value in enumerate(jobs)]
            for future in as_completed(futures):
                results.append(future.result())
                print(json.dumps(results[-1], sort_keys=True), flush=True)
        results.sort(key=lambda row: (row["experiment_id"], row["seed"]))
        with args.status.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader(); writer.writerows(results)
        passed = sum(bool(row["passed"]) for row in results)
        print(json.dumps({"status": "complete" if passed == len(results) else "failed",
                          "jobs": len(results), "passed": passed,
                          "status_file": str(args.status)}, indent=2))
        return 0 if passed == len(results) else 1
    except (OSError, ValueError, SweepError) as error:
        print(f"CAUSAL SWEEP FAILURE: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
