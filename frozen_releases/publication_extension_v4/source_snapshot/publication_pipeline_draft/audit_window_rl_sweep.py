#!/usr/bin/env python3
"""Audit all 50 matched external-window checkpoints without scoring returns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from publication_pipeline_draft.extension_release import (
    ExtensionReleaseError, verify_extension_release,
)
from publication_pipeline_draft.run_window_rl_sweep import verify_contract


class WindowAuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WindowAuditError(message)


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


def audit(contract_root: Path, status_path: Path, release_root: Path,
          repo_root: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"Audit output already exists: {output}")
    release = verify_extension_release(release_root, repo_root)
    contract, jobs = verify_contract(contract_root)
    require(contract.get("program_sha256") == release.get("program_sha256"),
            "Window contract and frozen extension use different programs.")
    statuses = read_csv(status_path)
    job_by_key = {(row["algorithm"], int(row["seed"])): row for row in jobs}
    status_by_key = {(row["algorithm"], int(row["seed"])): row
                     for row in statuses}
    require(len(job_by_key) == 50 and set(job_by_key) == set(status_by_key),
            "Status does not exactly match five algorithms by ten seeds.")
    require(all(boolean(row.get("passed")) for row in statuses),
            "At least one matched policy failed training or sanity gates.")
    try:
        import torch
    except ModuleNotFoundError as error:
        raise WindowAuditError("PyTorch is required for checkpoint audit.") from error

    records: list[dict[str, Any]] = []
    for key in sorted(job_by_key):
        job, status = job_by_key[key], status_by_key[key]
        run_dir = (repo_root / job["output_dir"]).resolve()
        checkpoint = run_dir / f"{job['CHECKPOINT_PREFIX']}_full.pt"
        report = run_dir / "sanity_no_holdout" / "sanity_report.json"
        gate_path = run_dir / "pretraining_behavior_gate.csv"
        require(checkpoint.is_file() and report.is_file() and gate_path.is_file(),
                f"Training evidence is incomplete: {run_dir}")
        report_value = json.loads(report.read_text(encoding="utf-8"))
        require(report_value.get("overall_pass") is True and
                report_value.get("publication_behavior_pass") is True,
                f"No-holdout sanity failed: {run_dir}")
        gate = read_csv(gate_path)
        require(bool(gate) and all(boolean(row.get("pass")) for row in gate),
                f"Behavior gate failed: {gate_path}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        architecture = payload.get("architecture")
        require(isinstance(architecture, dict),
                f"Checkpoint architecture missing: {checkpoint}")
        require(int(architecture.get("checkpoint_schema", 0)) in {5, 6},
                f"Unsupported checkpoint schema: {checkpoint}")
        expected = {
            "rl_algorithm": job["RL_ALGORITHM"],
            "policy_encoder": "lstm",
            "vine_feature_mode": "full",
            "cvar_observation_mode": "full",
            "cvar_reward_mode": "full",
            "pretrain_data_mode": "vine_synthetic",
            "run_finetune": True,
        }
        mismatches = {name: [architecture.get(name), value]
                      for name, value in expected.items()
                      if architecture.get(name) != value}
        require(not mismatches,
                f"Checkpoint metadata mismatch {checkpoint}: {mismatches}")
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
        require(bool(tensors) and all(bool(torch.isfinite(x).all()) for x in tensors),
                f"Checkpoint contains a non-finite tensor: {checkpoint}")
        parameter_count = int(architecture.get("parameter_count", 0))
        require(parameter_count > 0,
                f"Checkpoint parameter count is missing: {checkpoint}")
        update_count = int(payload.get("update_count", -1))
        total_actions = int(payload.get("total_actions", -1))
        require(update_count >= 0 and total_actions > 0,
                f"Checkpoint interaction/update counters are invalid: {checkpoint}")
        records.append({
            "window_id": contract["window_id"], "algorithm": key[0],
            "seed": key[1], "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_schema": architecture["checkpoint_schema"],
            "parameter_count": parameter_count,
            "update_count": update_count,
            "environment_interactions": total_actions,
            "tensor_count": len(tensors), "all_tensors_finite": True,
            "behavior_gate_pass": True, "sanity_pass": True,
            "duration_seconds": status.get("duration_seconds", ""),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        audit_path = temporary / "checkpoint_audit.csv"
        with audit_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        manifest = {
            "schema_version": 1,
            "status": "window_rl_sweep_audit_passed",
            "window_id": contract["window_id"], "job_count": 50,
            "algorithm_count": 5, "seeds_per_algorithm": 10,
            "all_checkpoint_tensors_finite": True,
            "all_checkpoint_metadata_match": True,
            "all_behavior_and_sanity_gates_pass": True,
            "window_contract_contents_sha256": sha256(
                contract_root / "CONTENTS.sha256"),
            "status_sha256": sha256(status_path),
            "extension_release_contents_sha256": release[
                "release_contents_sha256"],
            "checkpoint_audit_sha256": sha256(audit_path),
            "confirmatory_claim_permitted": False,
        }
        (temporary / "window_sweep_audit_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        checksum = [f"{sha256(path)}  {path.name}"
                    for path in sorted(temporary.iterdir()) if path.is_file()]
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
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.contract.resolve(), args.status.resolve(),
                       args.release.resolve(), args.repo_root.resolve(),
                       args.output)
    except (OSError, ValueError, json.JSONDecodeError, WindowAuditError,
            ExtensionReleaseError) as error:
        print(f"WINDOW RL AUDIT FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
