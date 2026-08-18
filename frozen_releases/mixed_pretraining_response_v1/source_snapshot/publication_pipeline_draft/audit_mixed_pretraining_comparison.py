#!/usr/bin/env python3
"""Audit ten new mixed full checkpoints and ten reused synthetic pretrained checkpoints."""

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

from publication_pipeline_draft.mixed_pretraining_protocol import (
    JOB_ENV_FIELDS, DoseProtocolError, load_contract, read_csv, require, sha256,
    verify_release,
)

MIXED_ARM = "mixed_pretraining_plus_historical_finetuning"
SYNTHETIC_ONLY_ARM = "synthetic_only_training"
SYNTHETIC_SOURCE = (
    "synthetic_100_unique_1000_presentations_no_policy_visible_dependence")


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def gate_failures(path: Path) -> list[str]:
    failures: list[str] = []
    for row in read_csv(path):
        require(math.isfinite(float(row.get("value", "nan"))),
                f"Non-finite behavior diagnostic: {path}")
        if not truthy(row.get("pass", "")):
            failures.append(row.get("metric", ""))
    require(not ({"gate_gross_mae", "max_position_limit_violation"} &
                 set(failures)), f"Hard-constraint gate failed: {path}")
    return failures


def finite_tensors(payload: Any, torch: Any) -> bool:
    tensors: list[Any] = []
    stack = [payload]
    while stack:
        value = stack.pop()
        if torch.is_tensor(value):
            tensors.append(value)
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return bool(tensors) and all(bool(torch.isfinite(value).all()) for value in tensors)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    require(not args.output.exists(), f"Mixed audit output exists: {args.output}")
    contract, contract_sha = load_contract(args.contract)
    release = verify_release(args.release, repo, args.jobs)
    require(release["contract_sha256"] == contract_sha,
            "Release and mixed contract differ.")
    jobs, statuses = read_csv(args.jobs), read_csv(args.status)
    job_by_key = {(row["experiment_id"], int(row["seed"])): row for row in jobs}
    status_by_key = {(row["experiment_id"], int(row["seed"])): row
                     for row in statuses}
    require(len(job_by_key) == len(status_by_key) == 10 and
            set(job_by_key) == set(status_by_key) and
            all(truthy(row["passed"]) for row in statuses),
            "All ten mixed job/status rows must pass.")
    prior_manifest_path = args.synthetic_audit / "synthetic_dose_audit_manifest.json"
    prior_table_path = args.synthetic_audit / "synthetic_dose_checkpoint_audit.csv"
    require(prior_manifest_path.is_file() and prior_table_path.is_file(),
            "Frozen synthetic-presentation audit is missing.")
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    require(prior_manifest.get("status") == "synthetic_dose_sweep_audit_passed" and
            prior_manifest.get("experiment_protocol") ==
            "synthetic_presentation_response_v2" and
            int(prior_manifest.get("job_count", -1)) == 20 and
            prior_manifest.get("all_checkpoint_tensors_finite") is True and
            prior_manifest.get("all_checkpoint_metadata_match") is True and
            prior_manifest.get("checkpoint_audit_sha256") == sha256(prior_table_path),
            "Prior synthetic checkpoint audit is not the frozen passed evidence.")
    prior_rows = [row for row in read_csv(prior_table_path)
                  if row["experiment_id"] == SYNTHETIC_SOURCE]
    prior_by_seed = {int(row["seed"]): row for row in prior_rows}
    prior_jobs = read_csv(args.synthetic_jobs)
    prior_job_by_seed = {int(row["seed"]): row for row in prior_jobs
                         if row["experiment_id"] == SYNTHETIC_SOURCE}
    require(set(prior_by_seed) == set(prior_job_by_seed) ==
            {int(value) for value in contract["seeds"]},
            "Prior synthetic no-visible checkpoint set is incomplete.")
    try:
        import torch
    except ModuleNotFoundError as error:
        raise DoseProtocolError("PyTorch is required for checkpoint audit.") from error

    expected_pretrain_actions = 1000 * int(contract["episode_length"])
    expected_full_actions = expected_pretrain_actions + 61 * int(contract["episode_length"])
    records: list[dict[str, Any]] = []
    for key in sorted(job_by_key):
        job = job_by_key[key]; seed = key[1]
        run = (repo / job["output_dir"]).resolve(); prefix = job["CHECKPOINT_PREFIX"]
        checkpoint = run / f"{prefix}_full.pt"
        pretrained = run / f"{prefix}_pretrained.pt"
        gate = run / "pretraining_behavior_gate.csv"
        require(checkpoint.is_file() and pretrained.is_file() and gate.is_file(),
                f"Mixed training artifacts are incomplete: {run}")
        failures = gate_failures(gate)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        pre_payload = torch.load(pretrained, map_location="cpu", weights_only=True)
        require(finite_tensors(payload, torch) and finite_tensors(pre_payload, torch),
                f"Mixed checkpoint contains non-finite tensors: {seed}")
        require(int(pre_payload.get("total_actions", -1)) == expected_pretrain_actions and
                int(payload.get("total_actions", -1)) == expected_full_actions,
                f"Mixed action accounting differs at seed {seed}.")
        architecture = payload.get("architecture")
        expected_architecture = {
            "rl_algorithm": "td3", "policy_encoder": "lstm",
            "vine_observation_mode": "zero", "vine_feature_mode": "zero",
            "cvar_observation_mode": "zero", "cvar_reward_mode": "full",
            "pretrain_data_mode": "mixed_historical_synthetic",
            "pretrain_behavior_gate_mode": "report_only", "run_finetune": True,
        }
        require(isinstance(architecture, dict) and all(
            architecture.get(name) == value
            for name, value in expected_architecture.items()),
            f"Mixed checkpoint metadata mismatch: {seed}")
        records.append({
            "arm_id": MIXED_ARM, "source_experiment_id": key[0], "seed": seed,
            "checkpoint_model": "full", "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint), "model_dir": str(run),
            "checkpoint_prefix": prefix, "all_tensors_finite": True,
            "behavior_gate_mode": "report_only", "behavior_gate_pass": not failures,
            "behavior_gate_failed_metrics": ";".join(failures),
            "pretrained_total_actions": int(pre_payload["total_actions"]),
            "selected_checkpoint_total_actions": int(payload["total_actions"]),
            **{field: job[field] for field in JOB_ENV_FIELDS},
        })

    for seed in sorted(prior_by_seed):
        audited = prior_by_seed[seed]; job = prior_job_by_seed[seed]
        checkpoint = Path(audited["pretrained_checkpoint"])
        require(checkpoint.is_file() and sha256(checkpoint) ==
                audited["pretrained_checkpoint_sha256"],
                f"Reused synthetic pretrained checkpoint changed: {seed}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(finite_tensors(payload, torch) and
                int(payload.get("total_actions", -1)) == expected_pretrain_actions,
                f"Reused synthetic pretrained checkpoint failed re-audit: {seed}")
        architecture = payload.get("architecture", {})
        require(architecture.get("pretrain_data_mode") == "vine_synthetic" and
                architecture.get("vine_feature_mode") == "zero" and
                architecture.get("cvar_observation_mode") == "zero" and
                architecture.get("run_finetune") is True,
                f"Reused synthetic architecture mismatch: {seed}")
        records.append({
            "arm_id": SYNTHETIC_ONLY_ARM,
            "source_experiment_id": SYNTHETIC_SOURCE, "seed": seed,
            "checkpoint_model": "pretrained", "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "model_dir": str(Path(audited["checkpoint"]).parent.resolve()),
            "checkpoint_prefix": job["CHECKPOINT_PREFIX"],
            "all_tensors_finite": True,
            "behavior_gate_mode": audited["behavior_gate_mode"],
            "behavior_gate_pass": truthy(audited["behavior_gate_pass"]),
            "behavior_gate_failed_metrics": audited["behavior_gate_failed_metrics"],
            "pretrained_total_actions": int(audited["pretrained_total_actions"]),
            "selected_checkpoint_total_actions": int(
                audited["pretrained_total_actions"]),
            **{field: job[field] for field in JOB_ENV_FIELDS},
        })
    require(len(records) == 20 and
            len({(row["arm_id"], row["seed"]) for row in records}) == 20,
            "Combined mixed comparison audit must contain 20 unique policies.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.",
                                      dir=args.output.parent))
    try:
        table = temporary / "mixed_pretraining_checkpoint_audit.csv"
        records.sort(key=lambda row: (row["arm_id"], row["seed"]))
        with table.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        manifest = {
            "schema_version": 1,
            "status": "mixed_pretraining_comparison_audit_passed",
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "terminal_same_holdout_training": True,
            "job_count": 20, "experiment_count": 2, "seeds_per_experiment": 10,
            "new_mixed_full_checkpoint_count": 10,
            "reused_synthetic_pretrained_checkpoint_count": 10,
            "all_checkpoint_tensors_finite": True,
            "all_checkpoint_metadata_match": True,
            "all_behavior_gate_enforcement_valid": True,
            "contract_sha256": contract_sha, "jobs_sha256": sha256(args.jobs),
            "status_sha256": sha256(args.status),
            "prior_synthetic_audit_sha256": sha256(prior_manifest_path),
            "checkpoint_audit_sha256": sha256(table),
        }
        (temporary / "mixed_pretraining_audit_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = sorted(path for path in temporary.iterdir() if path.is_file())
        (temporary / "CONTENTS.sha256").write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in files),
            encoding="ascii")
        os.replace(temporary, args.output); return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/mixed_pretraining_response_v1.json"))
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--synthetic-audit", required=True, type=Path)
    parser.add_argument("--synthetic-jobs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); args.repo_root = args.repo_root.resolve()
    args.contract = (args.repo_root / args.contract).resolve()
    for name in ("release", "jobs", "status", "synthetic_audit",
                 "synthetic_jobs", "output"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        result = audit(args)
    except (DoseProtocolError, OSError, ValueError, KeyError,
            json.JSONDecodeError) as error:
        print(f"MIXED PRETRAINING AUDIT FAILURE: {error}"); return 1
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
