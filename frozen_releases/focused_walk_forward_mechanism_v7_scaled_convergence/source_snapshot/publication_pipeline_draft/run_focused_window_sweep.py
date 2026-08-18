#!/usr/bin/env python3
"""Run one frozen 15-policy external-window mechanism sweep over GPUs."""

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

from publication_pipeline_draft.focused_window_training_protocol import (
    ENV_FIELDS, FocusedWindowError, sha256, validate_protocol, verify_contents,
)
from publication_pipeline_draft.extension_release import (
    ExtensionReleaseError, verify_extension_release,
)


class FocusedSweepError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FocusedSweepError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def attested_episode_counts(evidence: dict[str, Any]) -> dict[str, str]:
    """Return positive bundle-authoritative counts for both R subprocesses."""
    try:
        pretrain_episodes = int(evidence["pretrain_episodes"])
        finetune_episodes = int(evidence["finetune_episodes"])
    except (KeyError, TypeError, ValueError) as error:
        raise FocusedSweepError(
            "Window bundle attestation has invalid episode counts.") from error
    require(pretrain_episodes > 0 and finetune_episodes > 0,
            "Window bundle attestation has non-positive episode counts.")
    return {
        "PRETRAIN_EPISODES": str(pretrain_episodes),
        "FINETUNE_EPISODES": str(finetune_episodes),
    }


def verify_contract(root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    verify_contents(root)
    manifest_path = root / "focused_window_training_manifest.json"
    jobs_path = root / "focused_window_jobs.csv"
    protocol_path = root / "focused_mechanism_protocol.json"
    require(all(path.is_file() for path in (manifest_path, jobs_path, protocol_path)),
            "Focused window contract is incomplete.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("release_status") ==
            "frozen_focused_window_training_contract" and
            manifest.get("confirmatory_claim_permitted") is False,
            "Focused window contract has an invalid status.")
    protocol, digest = validate_protocol(protocol_path)
    require(digest == manifest.get("focused_protocol_sha256") and
            sha256(jobs_path) == manifest.get("jobs_sha256"),
            "Focused contract hashes do not reconcile.")
    jobs = read_csv(jobs_path)
    required = {"window_id", "panel_id", "experiment_id", "seed", "output_dir",
                *ENV_FIELDS}
    require(len(jobs) == 15 and required <= set(jobs[0]),
            "Focused job matrix is empty, incomplete, or not 3 by 5.")
    keys = {(row["experiment_id"], int(row["seed"])) for row in jobs}
    expected = {(item["experiment_id"], int(seed))
                for item in protocol["experiments"] for seed in protocol["seeds"]}
    require(keys == expected,
            "Focused job matrix differs from its snapshotted protocol.")
    return manifest, jobs


def run_job(row: dict[str, str], gpu: int, args: argparse.Namespace,
            cores: int, episode_counts: dict[str, str]) -> dict[str, Any]:
    repo = args.repo_root
    output = (repo / row["output_dir"]).resolve()
    require(not output.exists(), f"Immutable policy output exists: {output}")
    label = f"{row['experiment_id']}__{row['seed']}"
    environment = os.environ.copy()
    environment.update({field: row[field] for field in ENV_FIELDS})
    # Window bundles contain fewer historical trajectories than the original
    # full-sample configuration.  The attested bundle counts are authoritative
    # for both training and the subsequent sanity replay; allowing
    # run_with_config.r to fall back to config.yaml would silently change the
    # frozen window protocol.
    environment.update(episode_counts)
    environment.update({
        "TRAIN_SEED": row["seed"], "TRAIN_OUTPUT_DIR": str(output),
        "TRAIN_DEVICE": "cuda", "CUDA_VISIBLE_DEVICES": str(gpu),
        "VINE_SIM_CORES": str(cores), "OMP_NUM_THREADS": str(cores),
        "MKL_NUM_THREADS": str(cores),
        "RETICULATE_PYTHON": str(args.train_python),
        "POLICY_PYTHON": str(args.train_python),
        "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C", "TZ": "UTC",
    })
    start = time.monotonic()
    train_stdout = args.log_root / f"{label}.train.stdout.txt"
    train_stderr = args.log_root / f"{label}.train.stderr.txt"
    command = [str(args.rscript), "--vanilla", "run_with_config.r",
               str(args.config)]
    with train_stdout.open("wb") as stdout, train_stderr.open("wb") as stderr:
        train = subprocess.run(command, cwd=repo, env=environment,
                               stdout=stdout, stderr=stderr, check=False)
    sanity_exit: int | None = None
    if train.returncode == 0:
        sanity_stdout = args.log_root / f"{label}.sanity.stdout.txt"
        sanity_stderr = args.log_root / f"{label}.sanity.stderr.txt"
        with sanity_stdout.open("wb") as stdout, sanity_stderr.open("wb") as stderr:
            sanity = subprocess.run(
                [str(args.rscript), "--vanilla", "rl/training_sanity_check.r",
                 str(args.config)], cwd=repo, env=environment,
                stdout=stdout, stderr=stderr, check=False)
        sanity_exit = sanity.returncode
    checkpoint = output / f"{row['CHECKPOINT_PREFIX']}_full.pt"
    report_path = output / "sanity_no_holdout" / "sanity_report.json"
    sanity_structural_pass = False
    sanity_behavior_pass = False
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        sanity_structural_pass = bool(report.get("overall_pass"))
        sanity_behavior_pass = bool(report.get("publication_behavior_pass"))
    # Report-only mode preserves intended policies despite economic behavior
    # warnings, while the sanity script still fails non-finite tensors and hard
    # constraints.  The downstream checkpoint auditor rechecks these directly.
    report_only = row["PRETRAIN_BEHAVIOR_GATE_MODE"] == "report_only"
    passed = (train.returncode == 0 and sanity_exit == 0 and
              checkpoint.is_file() and report_path.is_file() and
              (sanity_structural_pass or report_only))
    return {
        "window_id": row["window_id"],
        "experiment_id": row["experiment_id"], "seed": int(row["seed"]),
        "gpu": gpu, "train_exit_code": train.returncode,
        "sanity_exit_code": sanity_exit,
        "checkpoint_exists": checkpoint.is_file(),
        "sanity_structural_pass": sanity_structural_pass,
        "sanity_behavior_pass": sanity_behavior_pass,
        "behavior_gate_mode": row["PRETRAIN_BEHAVIOR_GATE_MODE"],
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
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Validate the frozen release, contract, bundle, and episode counts without running jobs.")
    args = parser.parse_args()
    try:
        args.repo_root = args.repo_root.resolve()
        args.config = (args.repo_root / args.config).resolve()
        release = verify_extension_release(args.release.resolve(), args.repo_root)
        contract, jobs = verify_contract(args.contract.resolve())
        require(release.get("release_role") ==
                "focused_walk_forward_mechanism_v1" and
                release.get("focused_protocol_sha256") ==
                contract.get("focused_protocol_sha256") and
                release.get("program_sha256") == contract.get("program_sha256"),
                "Focused training contract and prospective release differ.")
        require(args.train_python.is_file() and args.rscript.is_file(),
                "Training Python/Rscript executable is missing.")
        require(not args.log_root.exists() and not args.status.exists(),
                "Immutable status/log output already exists.")
        bundle = args.repo_root / jobs[0]["SYNTHETIC_RETURNS_FILE"]
        bundle_manifest_path = bundle.parent / "synthetic_bundle_manifest.json"
        require(bundle.is_file() and bundle_manifest_path.is_file(),
                "Attested window synthetic bundle is missing.")
        evidence = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
        require(evidence.get("bundle_sha256") == sha256(bundle) and
                evidence.get("window_id") == jobs[0]["window_id"] and
                evidence.get("diagnostics_passed") is True,
                "Window synthetic bundle attestation failed.")
        episode_counts = attested_episode_counts(evidence)
        gpus = [int(item) for item in args.gpus.split(",")]
        require(gpus and len(gpus) == len(set(gpus)) and min(gpus) >= 0,
                "GPU identifiers are invalid.")
        for row in jobs:
            require(not (args.repo_root / row["output_dir"]).exists(),
                    f"Job output already exists: {row['output_dir']}")
        if args.preflight_only:
            print(json.dumps({
                "status": "focused_window_sweep_preflight_passed",
                "window_id": jobs[0]["window_id"],
                "jobs": len(jobs),
                "pretrain_episodes": int(
                    episode_counts["PRETRAIN_EPISODES"]),
                "finetune_episodes": int(
                    episode_counts["FINETUNE_EPISODES"]),
                "bundle_sha256": evidence["bundle_sha256"],
                "confirmatory_claim_permitted": False,
            }, indent=2, sort_keys=True))
            return 0
        args.log_root.mkdir(parents=True)
        args.status.parent.mkdir(parents=True, exist_ok=True)
        cores = max(1, args.cpu_cores // len(gpus))
        locks = {gpu: threading.Lock() for gpu in gpus}

        def assigned(item: tuple[int, dict[str, str]]) -> dict[str, Any]:
            index, row = item
            gpu = gpus[index % len(gpus)]
            with locks[gpu]:
                return run_job(row, gpu, args, cores, episode_counts)

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(assigned, item) for item in enumerate(jobs)]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
        results.sort(key=lambda row: (row["experiment_id"], row["seed"]))
        with args.status.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        passed = sum(bool(row["passed"]) for row in results)
        summary = {"status": "complete" if passed == 15 else "failed",
                   "jobs": 15, "passed": passed,
                   "confirmatory_claim_permitted": False}
        print(json.dumps(summary, indent=2))
        return 0 if passed == 15 else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            FocusedWindowError, FocusedSweepError,
            ExtensionReleaseError) as error:
        print(f"FOCUSED WINDOW SWEEP FAILURE: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
