#!/usr/bin/env python3
"""Run the frozen ten-policy mixed-pretraining sweep."""

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

from publication_pipeline_draft.mixed_pretraining_protocol import (
    JOB_ENV_FIELDS, DoseProtocolError, read_csv, require, sha256, verify_release,
)


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def run_job(row: dict[str, str], gpu: int, args: argparse.Namespace,
            log_root: Path, cores: int) -> dict[str, Any]:
    output = (args.repo_root / row["output_dir"]).resolve()
    require(not output.exists(), f"Immutable job output exists: {output}")
    bundle = (args.repo_root / row["SYNTHETIC_RETURNS_FILE"]).resolve()
    require(bundle.is_file() and sha256(bundle) == row["bundle_sha256"].lower(),
            f"Mixed bundle is missing or changed: {bundle}")
    environment = os.environ.copy()
    environment.update({field: row[field] for field in JOB_ENV_FIELDS})
    environment.update({
        "TRAIN_SEED": row["seed"], "TRAIN_OUTPUT_DIR": str(output),
        "TRAIN_DEVICE": "cuda", "CUDA_VISIBLE_DEVICES": str(gpu),
        "VINE_SIM_CORES": str(cores), "OMP_NUM_THREADS": str(cores),
        "MKL_NUM_THREADS": str(cores), "RETICULATE_PYTHON": str(args.train_python),
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
    prefix = row["CHECKPOINT_PREFIX"]
    checkpoint = output / f"{prefix}_full.pt"
    pretrained = output / f"{prefix}_pretrained.pt"
    gate = output / "pretraining_behavior_gate.csv"
    passed = (result.returncode == 0 and checkpoint.is_file() and
              pretrained.is_file() and gate.is_file())
    return {
        "experiment_id": row["experiment_id"], "seed": int(row["seed"]),
        "gpu": gpu, "exit_code": result.returncode,
        "duration_seconds": time.monotonic() - start,
        "output_dir": row["output_dir"], "checkpoint_exists": checkpoint.is_file(),
        "pretrained_checkpoint_exists": pretrained.is_file(),
        "gate_exists": gate.is_file(), "passed": passed,
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
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    args = parser.parse_args(); args.repo_root = args.repo_root.resolve()
    args.config = (args.repo_root / args.config).resolve()
    args.jobs = args.jobs.resolve(); args.release = args.release.resolve()
    args.train_python = args.train_python.resolve(); args.rscript = args.rscript.resolve()
    try:
        rows = read_csv(args.jobs)
        require(len(rows) == 10, "Mixed sweep requires ten jobs.")
        verify_release(args.release, args.repo_root, args.jobs)
        require(len({(row["experiment_id"], row["seed"]) for row in rows}) == 10,
                "Mixed job keys are duplicated.")
        require(all(row["PRETRAIN_DATA_MODE"] == "mixed_historical_synthetic" and
                    row["PRETRAIN_EPISODES"] == "1000" and
                    row["RUN_FINETUNE"] == "true" and
                    row["VINE_FEATURE_MODE"] == "zero" and
                    row["CVAR_OBSERVATION_MODE"] == "zero" for row in rows),
                "Jobs do not preserve the frozen mixed design.")
        require(not args.status.exists() and not args.log_root.exists(),
                "Mixed status/log output already exists.")
        require(args.train_python.is_file() and args.rscript.is_file(),
                "Training Python or Rscript was not found.")
        gpus = [int(value) for value in args.gpus.split(",")]
        require(bool(gpus) and len(gpus) == len(set(gpus)) and min(gpus) >= 0,
                "GPU identifiers must be distinct non-negative integers.")
        require(args.cpu_cores >= len(gpus), "Fewer CPU cores than GPU workers.")
        for row in rows:
            require(not (args.repo_root / row["output_dir"]).exists(),
                    f"Job output already exists: {row['output_dir']}")
        args.log_root.mkdir(parents=True); args.status.parent.mkdir(parents=True,
                                                                    exist_ok=True)
        cores = max(1, args.cpu_cores // len(gpus))
        locks = {gpu: threading.Lock() for gpu in gpus}

        def assigned(index_row: tuple[int, dict[str, str]]) -> dict[str, Any]:
            index, row = index_row; gpu = gpus[index % len(gpus)]
            with locks[gpu]:
                return run_job(row, gpu, args, args.log_root, cores)

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(assigned, value) for value in enumerate(rows)]
            for future in as_completed(futures):
                row = future.result(); results.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
        results.sort(key=lambda row: (row["experiment_id"], row["seed"]))
        with args.status.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader(); writer.writerows(results)
        passed = sum(truthy(row["passed"]) for row in results)
        summary = {"status": "complete" if passed == 10 else "failed",
                   "jobs": 10, "passed": passed, "status_file": str(args.status)}
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if passed == 10 else 1
    except (DoseProtocolError, OSError, ValueError, KeyError) as error:
        print(f"MIXED PRETRAINING SWEEP FAILURE: {error}"); return 1


if __name__ == "__main__":
    raise SystemExit(main())
