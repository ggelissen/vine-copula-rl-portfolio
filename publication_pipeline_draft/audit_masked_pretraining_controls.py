#!/usr/bin/env python3
"""Fail-closed audit of terminal masked pretraining-control checkpoints."""

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

from publication_pipeline_draft.masked_pretraining_controls_protocol import (
    DoseProtocolError, load_contract, read_csv, require, sha256, verify_release,
)


def boolean(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def gate_failures(path: Path) -> list[str]:
    failures: list[str] = []
    for row in read_csv(path):
        require(math.isfinite(float(row.get("value", "nan"))),
                f"Non-finite behavior diagnostic: {path}")
        if not boolean(row.get("pass", "")):
            failures.append(row.get("metric", ""))
    require(not ({"gate_gross_mae", "max_position_limit_violation"} & set(failures)),
            f"Hard-constraint behavior gate failed: {path}")
    return failures


def audit(repo: Path, contract_path: Path, release: Path, jobs_path: Path,
          status_path: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"Masked-control audit output exists: {output}")
    contract, contract_sha = load_contract(contract_path)
    release_manifest = verify_release(release, repo, jobs_path)
    require(release_manifest["contract_sha256"] == contract_sha,
            "Release and live contract differ.")
    jobs, statuses = read_csv(jobs_path), read_csv(status_path)
    require(len(jobs) == len(statuses) == 20,
            "Masked-control audit requires 20 job/status rows.")
    job_by_key = {(row["experiment_id"], int(row["seed"])): row for row in jobs}
    status_by_key = {(row["experiment_id"], int(row["seed"])): row
                     for row in statuses}
    require(len(job_by_key) == 20 and set(job_by_key) == set(status_by_key),
            "Job/status keys differ or are duplicated.")
    require(all(boolean(row["passed"]) for row in statuses),
            "At least one masked-control training job failed.")
    try:
        import torch
    except ModuleNotFoundError as error:
        raise DoseProtocolError("PyTorch is required for checkpoint audit.") from error

    expected_pretrain_actions = 1000 * int(contract["episode_length"])
    expected_full_actions = expected_pretrain_actions + int(
        contract["finetune_episodes"]) * int(contract["episode_length"])
    records: list[dict[str, Any]] = []
    update_counts: set[tuple[int, int]] = set()
    for key in sorted(job_by_key):
        job = job_by_key[key]; run = (repo / job["output_dir"]).resolve()
        prefix = job["CHECKPOINT_PREFIX"]
        pretrained = run / f"{prefix}_pretrained.pt"
        checkpoint = run / f"{prefix}_full.pt"
        gate = run / "pretraining_behavior_gate.csv"
        require(pretrained.is_file() and checkpoint.is_file() and gate.is_file(),
                f"Masked-control artifacts are incomplete: {run}")
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
            "vine_observation_mode": "zero", "vine_feature_mode": "zero",
            "cvar_observation_mode": "zero", "cvar_reward_mode": "full",
            "pretrain_data_mode": job["PRETRAIN_DATA_MODE"],
            "pretrain_behavior_gate_mode": "report_only", "run_finetune": True,
        }
        mismatches = {name: (architecture.get(name), value)
                      for name, value in expected.items()
                      if architecture.get(name) != value}
        require(not mismatches, f"Checkpoint metadata mismatch {key}: {mismatches}")
        tensors: list[Any] = []; stack: list[Any] = list(payloads.values())
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
        pre_updates = int(payloads["pretrained"].get("update_count", -1))
        full_updates = int(payloads["full"].get("update_count", -1))
        require(pre_updates > 0 and full_updates >= pre_updates,
                f"Invalid update accounting at {key}.")
        update_counts.add((pre_updates, full_updates))
        records.append({
            "experiment_id": key[0], "seed": key[1],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "pretrained_checkpoint": str(pretrained.resolve()),
            "pretrained_checkpoint_sha256": sha256(pretrained),
            "checkpoint_schema": architecture.get("checkpoint_schema"), **expected,
            "pretrain_episode_presentations": 1000,
            "pretrained_total_actions": int(payloads["pretrained"]["total_actions"]),
            "full_total_actions": int(payloads["full"]["total_actions"]),
            "pretrained_update_count": pre_updates,
            "full_update_count": full_updates,
            "bundle_sha256": job["bundle_sha256"], "all_tensors_finite": True,
            "behavior_gate_mode": "report_only", "behavior_gate_pass": not failures,
            "behavior_gate_failed_metrics": ";".join(failures),
        })
    require(len(update_counts) == 1,
            "Controls did not receive identical gradient-update budgets.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        table = temporary / "synthetic_dose_checkpoint_audit.csv"
        with table.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        manifest = {
            "schema_version": 1,
            "status": "synthetic_dose_sweep_audit_passed",
            "experiment_protocol": "terminal_masked_pretraining_controls_v1",
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False, "terminal_hpc_experiment": True,
            "job_count": 20, "experiment_count": 2, "seeds_per_experiment": 10,
            "all_checkpoint_tensors_finite": True,
            "all_checkpoint_metadata_match": True,
            "all_behavior_gate_enforcement_valid": True,
            "identical_update_budget": True,
            "economic_behavior_pass_count": sum(
                bool(row["behavior_gate_pass"]) for row in records),
            "pretrain_episode_presentations": 1000,
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
        os.replace(temporary, output); return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/masked_pretraining_controls_v1.json"))
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); repo = args.repo_root.resolve()
    try:
        result = audit(repo, (repo / args.contract).resolve(), args.release.resolve(),
                       args.jobs.resolve(), args.status.resolve(), args.output.resolve())
    except (DoseProtocolError, OSError, ValueError, KeyError) as error:
        print(f"MASKED PRETRAINING CONTROL AUDIT FAILURE: {error}"); return 1
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
