#!/usr/bin/env python3
"""Fail-closed checkpoint and behavioral audit for a causal sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from publication_pipeline_draft.causal_ablation_protocol import ENV_FIELDS


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boolean(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def validate_behavior_gate(gate: list[dict[str, str]], gate_mode: str,
                           label: str) -> list[str]:
    """Validate intent-to-train diagnostics without selecting good controls."""
    require(bool(gate), f"Behavioral gate is empty: {label}")
    require(gate_mode in {"strict", "report_only"},
            f"Invalid behavior-gate mode: {gate_mode}")
    failures = [row.get("metric", "") for row in gate
                if not boolean(row.get("pass", ""))]
    structural_metrics = {"gate_gross_mae", "max_position_limit_violation"}
    nonfinite = []
    for row in gate:
        try:
            finite = math.isfinite(float(row.get("value", "nan")))
        except ValueError:
            finite = False
        if not finite:
            nonfinite.append(row.get("metric", ""))
    require(not nonfinite, f"Behavioral diagnostics are non-finite: {label}")
    require(not (set(failures) & structural_metrics),
            f"Hard-constraint gate failed: {label}")
    require(gate_mode == "report_only" or not failures,
            f"Strict behavioral gate failed: {label}")
    return failures


def audit(jobs_path: Path, status_path: Path, repo_root: Path,
          output: Path, operational_merge_manifest: Path | None = None) -> dict[str, Any]:
    require(not output.exists(), f"Audit output already exists: {output}")
    jobs = read_csv(jobs_path); statuses = read_csv(status_path)
    require(bool(jobs) and bool(statuses), "Job matrix/status is empty.")
    job_by_key = {(row["experiment_id"], int(row["seed"])): row for row in jobs}
    status_by_key = {(row["experiment_id"], int(row["seed"])): row for row in statuses}
    require(len(job_by_key) == len(jobs), "Job matrix keys are duplicated.")
    require(set(job_by_key) == set(status_by_key),
            "Status rows do not exactly match the preregistered job matrix.")
    require(all(boolean(row["passed"]) for row in statuses),
            "At least one preregistered training job failed.")
    experiments: dict[str, set[int]] = {}
    for experiment, seed in job_by_key:
        experiments.setdefault(experiment, set()).add(seed)
    require(all(len(seeds) == 10 for seeds in experiments.values()),
            "Every experiment must contain exactly ten seeds.")

    try:
        import torch
    except ModuleNotFoundError as error:
        raise AuditError("PyTorch is required to inspect checkpoint tensors.") from error
    records: list[dict[str, Any]] = []
    feedforward_counts: list[int] = []
    feedforward_targets: list[int] = []
    for key in sorted(job_by_key):
        job = job_by_key[key]
        run_dir = (repo_root / job["output_dir"]).resolve()
        checkpoint = run_dir / f"{job['CHECKPOINT_PREFIX']}_full.pt"
        gate_path = run_dir / "pretraining_behavior_gate.csv"
        require(checkpoint.is_file(), f"Checkpoint missing: {checkpoint}")
        gate = read_csv(gate_path)
        gate_mode = job.get("PRETRAIN_BEHAVIOR_GATE_MODE", "strict")
        failed_gate_metrics = validate_behavior_gate(
            gate, gate_mode, str(gate_path))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        architecture = payload.get("architecture")
        require(isinstance(architecture, dict), f"Architecture missing: {checkpoint}")
        architecture = dict(architecture)
        # The 70 successful v2 checkpoints predate this provenance field and
        # necessarily used the default strict gate.  The operational merger
        # permits that default only for strict-gate survivors.
        architecture.setdefault("pretrain_behavior_gate_mode", "strict")
        require(int(architecture.get("checkpoint_schema", 0)) in {5, 6},
                f"Unsupported checkpoint schema: {checkpoint}")
        expected = {
            "rl_algorithm": job["RL_ALGORITHM"],
            "policy_encoder": job["POLICY_ENCODER"],
            "vine_feature_mode": job["VINE_FEATURE_MODE"],
            "cvar_observation_mode": job["CVAR_OBSERVATION_MODE"],
            "cvar_reward_mode": job["CVAR_REWARD_MODE"],
            "pretrain_data_mode": job["PRETRAIN_DATA_MODE"],
            "pretrain_behavior_gate_mode": gate_mode,
            "run_finetune": boolean(job["RUN_FINETUNE"]),
        }
        mismatches = {field: [architecture.get(field), value]
                      for field, value in expected.items()
                      if architecture.get(field) != value}
        require(not mismatches, f"Checkpoint metadata mismatch {checkpoint}: {mismatches}")
        tensors = []
        stack = [payload]
        while stack:
            value = stack.pop()
            if torch.is_tensor(value):
                tensors.append(value)
            elif isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
        require(bool(tensors) and all(bool(torch.isfinite(value).all()) for value in tensors),
                f"Checkpoint contains a non-finite tensor: {checkpoint}")
        count = int(architecture.get("parameter_count", 0))
        require(count > 0, "Checkpoint lacks a positive trainable parameter count.")
        update_count = int(payload.get("update_count", -1))
        total_actions = int(payload.get("total_actions", -1))
        require(update_count >= 0 and total_actions > 0,
                "Checkpoint lacks valid update/interaction counters.")
        target = architecture.get("capacity_target_parameter_count")
        if job["POLICY_ENCODER"] == "mlp":
            require(count > 0 and isinstance(target, int) and target > 0,
                    "Feedforward checkpoint lacks capacity-match evidence.")
            require(abs(count - target) / target <= 0.05,
                    "Feedforward parameter count is not within 5% of recurrent TD3.")
            feedforward_counts.append(count); feedforward_targets.append(target)
        records.append({
            "experiment_id": job["experiment_id"], "seed": int(job["seed"]),
            "checkpoint": str(checkpoint.relative_to(repo_root)),
            "sha256": sha256(checkpoint), "size_bytes": checkpoint.stat().st_size,
            "checkpoint_schema": architecture.get("checkpoint_schema"),
            **expected, "parameter_count": count,
            "update_count": update_count,
            "environment_interactions": total_actions,
            "capacity_target_parameter_count": target,
            "tensor_count": len(tensors), "all_tensors_finite": True,
            "behavior_gate_pass": not failed_gate_metrics,
            "behavior_gate_mode": gate_mode,
            "behavior_gate_failed_metrics": ";".join(failed_gate_metrics),
            "operational_source": job.get("operational_source", "single_revision"),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        with (temporary / "checkpoint_audit.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        merge_evidence = None
        if operational_merge_manifest is not None:
            require(operational_merge_manifest.is_file(),
                    "Operational merge manifest was not found.")
            merge_evidence = json.loads(operational_merge_manifest.read_text(
                encoding="utf-8"))
            require(merge_evidence.get("status") in {
                    "complete_70_v2_plus_60_v3_operational_merge",
                    "complete_70_v2_plus_31_v3_plus_29_v4_operational_merge"},
                    "Operational merge manifest has the wrong status.")
            require(merge_evidence.get("combined_jobs_sha256") == sha256(jobs_path) and
                    merge_evidence.get("combined_status_sha256") == sha256(status_path),
                    "Operational merge manifest is not bound to the audited inputs.")
        manifest = {
            "schema_version": 1,
            "status": "causal_sweep_audit_passed",
            "job_count": len(records), "experiment_count": len(experiments),
            "seeds_per_experiment": 10, "all_checkpoint_tensors_finite": True,
            "all_behavior_gates_pass": all(
                bool(row["behavior_gate_pass"]) for row in records),
            "all_behavior_gate_enforcement_valid": True,
            "all_checkpoint_metadata_match": True,
            "feedforward_capacity_within_5_percent": bool(feedforward_counts),
            "jobs_sha256": sha256(jobs_path), "status_sha256": sha256(status_path),
            "mixed_revision_carry_forward": merge_evidence is not None,
            "operational_merge_manifest_sha256": (
                sha256(operational_merge_manifest)
                if operational_merge_manifest is not None else None),
            "v2_carried_count": (merge_evidence.get("v2_carried_count")
                                 if merge_evidence else 0),
            "v3_retry_count": (merge_evidence.get("v3_retry_count")
                                 if merge_evidence else 0),
            "v3_carried_count": (merge_evidence.get("v3_carried_count")
                                  if merge_evidence else 0),
            "v4_retry_count": (merge_evidence.get("v4_retry_count")
                                if merge_evidence else 0),
            "confirmatory_claim_permitted": False,
            "claim_limit": "post-holdout explanatory or external-development evidence only",
        }
        (temporary / "causal_sweep_audit_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "READ_ONLY_AUDIT.txt").write_text(
            "Freeze this audit with the job contract before any external test batch.\n",
            encoding="utf-8")
        os.replace(temporary, output)
        return manifest
    except Exception:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operational-merge-manifest", type=Path)
    args = parser.parse_args()
    if args.operational_merge_manifest is not None:
        args.operational_merge_manifest = args.operational_merge_manifest.resolve()
    try:
        manifest = audit(args.jobs, args.status, args.repo_root.resolve(), args.output,
                         args.operational_merge_manifest)
    except (AuditError, OSError, ValueError) as error:
        print(f"CAUSAL SWEEP AUDIT FAILURE: {error}")
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
