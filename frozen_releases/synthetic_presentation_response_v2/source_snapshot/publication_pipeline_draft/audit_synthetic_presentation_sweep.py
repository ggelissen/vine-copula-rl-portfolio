#!/usr/bin/env python3
"""Fail-closed audit of the 100-unique/1000-presentation checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from publication_pipeline_draft.synthetic_presentation_protocol import (
    DoseProtocolError, load_contract, read_csv, require, sha256, verify_release,
)


def boolean(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def gate_failures(path: Path) -> list[str]:
    rows = read_csv(path)
    failures: list[str] = []
    for row in rows:
        require(math.isfinite(float(row.get("value", "nan"))),
                f"Non-finite behavior diagnostic: {path}")
        if not boolean(row.get("pass", "")):
            failures.append(row.get("metric", ""))
    require(not ({"gate_gross_mae", "max_position_limit_violation"} & set(failures)),
            f"Hard-constraint behavior gate failed: {path}")
    return failures


def audit(repo: Path, contract_path: Path, release: Path, jobs_path: Path,
          status_path: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"Presentation audit output exists: {output}")
    contract, contract_sha = load_contract(contract_path)
    release_manifest = verify_release(release, repo, jobs_path)
    require(release_manifest["contract_sha256"] == contract_sha,
            "Release and live contract differ.")
    jobs = read_csv(jobs_path); statuses = read_csv(status_path)
    require(len(jobs) == len(statuses) == 20,
            "Presentation audit requires 20 job/status rows.")
    job_by_key = {(row["experiment_id"], int(row["seed"])): row for row in jobs}
    status_by_key = {(row["experiment_id"], int(row["seed"])): row
                     for row in statuses}
    require(len(job_by_key) == 20 and set(job_by_key) == set(status_by_key),
            "Job/status keys differ or are duplicated.")
    require(all(boolean(row["passed"]) for row in statuses),
            "At least one presentation training job failed.")
    try:
        import torch
    except ModuleNotFoundError as error:
        raise DoseProtocolError("PyTorch is required for checkpoint audit.") from error

    expected_pretrain_actions = 1000 * int(contract["episode_length"])
    expected_full_actions = expected_pretrain_actions + int(
        contract["finetune_episodes"]) * int(contract["episode_length"])
    records: list[dict[str, Any]] = []
    for key in sorted(job_by_key):
        job = job_by_key[key]
        run = (repo / job["output_dir"]).resolve()
        prefix = job["CHECKPOINT_PREFIX"]
        pretrained = run / f"{prefix}_pretrained.pt"
        checkpoint = run / f"{prefix}_full.pt"
        gate = run / "pretraining_behavior_gate.csv"
        require(pretrained.is_file() and checkpoint.is_file() and gate.is_file(),
                f"Presentation artifacts are incomplete: {run}")
        failures = gate_failures(gate)
        payloads = {
            "pretrained": torch.load(pretrained, map_location="cpu", weights_only=True),
            "full": torch.load(checkpoint, map_location="cpu", weights_only=True),
        }
        require(int(payloads["pretrained"].get("total_actions", -1)) ==
                expected_pretrain_actions,
                f"Pretraining exposure differs from 1000x24 at {key}.")
        require(int(payloads["full"].get("total_actions", -1)) == expected_full_actions,
                f"Full exposure differs from 1000x24 + 61x24 at {key}.")
        architecture = payloads["full"].get("architecture")
        require(isinstance(architecture, dict), f"Architecture missing: {key}")
        expected = {
            "rl_algorithm": job["RL_ALGORITHM"],
            "policy_encoder": job["POLICY_ENCODER"],
            "vine_observation_mode": job["VINE_OBSERVATION_MODE"],
            "vine_feature_mode": job["VINE_FEATURE_MODE"],
            "cvar_observation_mode": job["CVAR_OBSERVATION_MODE"],
            "cvar_reward_mode": job["CVAR_REWARD_MODE"],
            "pretrain_data_mode": job["PRETRAIN_DATA_MODE"],
            "pretrain_behavior_gate_mode": job["PRETRAIN_BEHAVIOR_GATE_MODE"],
            "run_finetune": True,
        }
        mismatches = {name: (architecture.get(name), value)
                      for name, value in expected.items()
                      if architecture.get(name) != value}
        require(not mismatches, f"Checkpoint metadata mismatch {key}: {mismatches}")
        tensors = []
        stack: list[Any] = list(payloads.values())
        while stack:
            value = stack.pop()
            if torch.is_tensor(value):
                tensors.append(value)
            elif isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
        require(bool(tensors) and all(bool(torch.isfinite(value).all())
                                      for value in tensors),
                f"Checkpoint contains a non-finite tensor: {key}")
        records.append({
            "experiment_id": key[0], "seed": key[1],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "pretrained_checkpoint": str(pretrained.resolve()),
            "pretrained_checkpoint_sha256": sha256(pretrained),
            "checkpoint_schema": architecture.get("checkpoint_schema"), **expected,
            "synthetic_unique_episode_count": 100,
            "pretrain_episode_presentations": 1000,
            "synthetic_repetition_count": 10,
            "pretrain_random_exploration_steps": int(
                job["PRETRAIN_RANDOM_EXPLORATION_STEPS"]),
            "pretrain_noise_decay": float(job["PRETRAIN_NOISE_DECAY"]),
            "pretrained_total_actions": int(payloads["pretrained"]["total_actions"]),
            "full_total_actions": int(payloads["full"]["total_actions"]),
            "pretrained_update_count": int(payloads["pretrained"]["update_count"]),
            "full_update_count": int(payloads["full"]["update_count"]),
            "bundle_sha256": job["bundle_sha256"],
            "all_tensors_finite": True,
            "behavior_gate_mode": job["PRETRAIN_BEHAVIOR_GATE_MODE"],
            "behavior_gate_pass": not failures,
            "behavior_gate_failed_metrics": ";".join(failures),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        table = temporary / "synthetic_dose_checkpoint_audit.csv"
        with table.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        manifest = {
            "schema_version": 1,
            # This existing authorization status is intentionally reused by the
            # isolated evaluator.  The v2-specific fields and hashes below are
            # checked by the v2 replay driver before authorization is granted.
            "status": "synthetic_dose_sweep_audit_passed",
            "experiment_protocol": "synthetic_presentation_response_v2",
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "job_count": 20, "experiment_count": 2, "seeds_per_experiment": 10,
            "all_checkpoint_tensors_finite": True,
            "all_checkpoint_metadata_match": True,
            "all_behavior_gate_enforcement_valid": True,
            "economic_behavior_pass_count": sum(
                bool(row["behavior_gate_pass"]) for row in records),
            "pretrain_episode_presentations": 1000,
            "synthetic_unique_episode_count": 100,
            "synthetic_repetition_count": 10,
            "historical_finetune_episode_count": 61,
            "jobs_sha256": sha256(jobs_path), "status_sha256": sha256(status_path),
            "checkpoint_audit_sha256": sha256(table),
            "release_contents_sha256": sha256(release / "CONTENTS.sha256"),
        }
        (temporary / "synthetic_dose_audit_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = sorted(path for path in temporary.iterdir() if path.is_file())
        (temporary / "CONTENTS.sha256").write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in files),
            encoding="ascii")
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/synthetic_presentation_response_v2.json"))
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    try:
        result = audit(repo, (repo / args.contract).resolve(), args.release.resolve(),
                       args.jobs.resolve(), args.status.resolve(), args.output.resolve())
    except (DoseProtocolError, OSError, ValueError, KeyError) as error:
        print(f"SYNTHETIC PRESENTATION AUDIT FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
