#!/usr/bin/env python3
"""Audit all 15 focused external-window checkpoints without scoring returns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from publication_pipeline_draft.run_focused_window_sweep import verify_contract


class FocusedAuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FocusedAuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def validate_behavior_gate(rows: list[dict[str, str]], mode: str,
                           label: str) -> list[str]:
    require(rows, f"Behavior gate is empty: {label}")
    failures: list[str] = []
    structural = {"gate_gross_mae", "max_position_limit_violation"}
    for row in rows:
        metric = row.get("metric", "")
        try:
            finite = math.isfinite(float(row.get("value", "nan")))
        except ValueError:
            finite = False
        require(finite, f"Non-finite behavior metric {metric}: {label}")
        if not boolean(row.get("pass", "")):
            failures.append(metric)
    require(not (set(failures) & structural),
            f"Hard-constraint behavior gate failed: {label}")
    require(mode == "report_only" or not failures,
            f"Strict behavior gate failed: {label}")
    return failures


def audit(contract_root: Path, status_path: Path, repo_root: Path,
          output: Path) -> dict[str, Any]:
    require(not output.exists(), f"Audit output already exists: {output}")
    contract, jobs = verify_contract(contract_root)
    statuses = read_csv(status_path)
    job_by_key = {(row["experiment_id"], int(row["seed"])): row for row in jobs}
    status_by_key = {(row["experiment_id"], int(row["seed"])): row
                     for row in statuses}
    require(len(job_by_key) == 15 and set(job_by_key) == set(status_by_key),
            "Sweep status does not exactly match the 3-by-5 contract.")
    require(all(boolean(row.get("passed")) for row in statuses),
            "At least one focused policy failed training evidence collection.")
    try:
        import torch
    except ModuleNotFoundError as error:
        raise FocusedAuditError("PyTorch is required for checkpoint audit.") from error

    records: list[dict[str, Any]] = []
    for key in sorted(job_by_key):
        job, status = job_by_key[key], status_by_key[key]
        run_dir = (repo_root / job["output_dir"]).resolve()
        checkpoint = run_dir / f"{job['CHECKPOINT_PREFIX']}_full.pt"
        gate_path = run_dir / "pretraining_behavior_gate.csv"
        report_path = run_dir / "sanity_no_holdout" / "sanity_report.json"
        require(checkpoint.is_file() and gate_path.is_file() and report_path.is_file(),
                f"Focused training evidence is incomplete: {run_dir}")
        mode = job["PRETRAIN_BEHAVIOR_GATE_MODE"]
        gate_failures = validate_behavior_gate(read_csv(gate_path), mode,
                                               str(gate_path))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(report.get("vine_feature_mode") == job["VINE_FEATURE_MODE"] and
                report.get("cvar_observation_mode") ==
                job["CVAR_OBSERVATION_MODE"] and
                report.get("cvar_reward_mode") == job["CVAR_REWARD_MODE"],
                f"Sanity report mode mismatch: {report_path}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        architecture = payload.get("architecture")
        require(isinstance(architecture, dict),
                f"Checkpoint architecture missing: {checkpoint}")
        expected = {
            "rl_algorithm": job["RL_ALGORITHM"],
            "policy_encoder": job["POLICY_ENCODER"],
            "vine_feature_mode": job["VINE_FEATURE_MODE"],
            "cvar_observation_mode": job["CVAR_OBSERVATION_MODE"],
            "cvar_reward_mode": job["CVAR_REWARD_MODE"],
            "pretrain_data_mode": job["PRETRAIN_DATA_MODE"],
            "pretrain_behavior_gate_mode": mode,
            "run_finetune": boolean(job["RUN_FINETUNE"]),
        }
        observed = dict(architecture)
        observed.setdefault("pretrain_behavior_gate_mode", "strict")
        mismatches = {name: [observed.get(name), value]
                      for name, value in expected.items()
                      if observed.get(name) != value}
        require(not mismatches,
                f"Checkpoint metadata mismatch {checkpoint}: {mismatches}")
        tensors: list[Any] = []
        stack: list[Any] = [payload]
        while stack:
            value = stack.pop()
            if torch.is_tensor(value):
                tensors.append(value)
            elif isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
        require(tensors and all(bool(torch.isfinite(value).all())
                                for value in tensors),
                f"Checkpoint contains a non-finite tensor: {checkpoint}")
        parameter_count = int(observed.get("parameter_count", 0))
        require(parameter_count > 0 and int(payload.get("update_count", -1)) >= 0 and
                int(payload.get("total_actions", -1)) > 0,
                f"Checkpoint counters or parameter count are invalid: {checkpoint}")
        sanity_warnings = list(report.get("warnings", []))
        require(mode == "report_only" or not sanity_warnings,
                f"Strict sanity warnings found: {report_path}")
        records.append({
            "window_id": contract["window_id"],
            "experiment_id": key[0], "seed": key[1],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_schema": observed.get("checkpoint_schema"),
            "parameter_count": parameter_count,
            "update_count": int(payload.get("update_count", -1)),
            "environment_interactions": int(payload.get("total_actions", -1)),
            "tensor_count": len(tensors), "all_tensors_finite": True,
            "behavior_gate_mode": mode,
            "behavior_gate_pass": not gate_failures,
            "behavior_gate_failed_metrics": ";".join(gate_failures),
            "sanity_behavior_pass": boolean(report.get("publication_behavior_pass")),
            "sanity_warning_count": len(sanity_warnings),
            "duration_seconds": status.get("duration_seconds", ""),
            **expected,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        audit_path = temporary / "focused_checkpoint_audit.csv"
        with audit_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        manifest = {
            "schema_version": 1,
            "status": "focused_window_sweep_audit_passed",
            "window_id": contract["window_id"],
            "job_count": 15, "experiment_count": 3,
            "seeds_per_experiment": 5,
            "protocol_eligible_policy_count": 15,
            "all_checkpoint_tensors_finite": True,
            "all_checkpoint_metadata_match": True,
            "all_behavior_gate_enforcement_valid": True,
            "all_economic_diagnostics_pass_count": sum(
                bool(row["behavior_gate_pass"]) for row in records),
            "report_only_included_count": sum(
                not bool(row["behavior_gate_pass"]) for row in records),
            "contract_contents_sha256": sha256(contract_root / "CONTENTS.sha256"),
            "status_sha256": sha256(status_path),
            "checkpoint_audit_sha256": sha256(audit_path),
            "confirmatory_claim_permitted": False,
        }
        (temporary / "focused_sweep_audit_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        checksum = [f"{sha256(path)}  {path.name}" for path in
                    sorted(temporary.iterdir()) if path.is_file() and
                    path.name != "CONTENTS.sha256"]
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(checksum) + "\n", encoding="ascii")
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.contract.resolve(), args.status.resolve(),
                       args.repo_root.resolve(), args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            FocusedAuditError) as error:
        print(f"FOCUSED WINDOW AUDIT FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
