#!/usr/bin/env python3
"""Merge the exact 70 successful v2 jobs with exact 60 same-seed v3 retries.

This is deliberately narrower than a generic status merger.  It permits no
seed substitution, requires every v2 survivor to have passed its strict gate,
and permits the two job contracts to differ only in output/provenance fields
and the disclosed strict-to-report-only gate-policy revision.
"""

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
from publication_pipeline_draft.extension_release import (
    ExtensionReleaseError,
    verify_frozen_extension_integrity,
)


class OperationalMergeError(RuntimeError):
    pass


EXPECTED_ORIGINAL_PASSED = 70
EXPECTED_RETRIED = 60
IGNORED_JOB_FIELDS = {"output_dir", "contract_sha256"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OperationalMergeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file(), f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def keyed(rows: list[dict[str, str]], label: str) -> dict[tuple[str, str], dict[str, str]]:
    result = {(row["experiment_id"], row["seed"]): row for row in rows}
    require(len(result) == len(rows), f"{label} contains duplicate experiment/seed keys.")
    return result


def verify_release(release: Path, jobs: Path) -> dict[str, Any]:
    manifest = verify_frozen_extension_integrity(release)
    require(manifest.get("causal_jobs_sha256") == sha256(jobs),
            f"Frozen release is not bound to job matrix: {jobs}")
    return {"path": str(release.resolve()),
            "contents_sha256": manifest["release_contents_sha256"],
            "jobs_sha256": sha256(jobs)}


def verify_completed_job(repo: Path, job: dict[str, str],
                         status: dict[str, str], strict_gate_required: bool) -> None:
    require(status.get("exit_code") == "0" and truth(status.get("passed", "")) and
            truth(status.get("checkpoint_exists", "")) and
            truth(status.get("gate_exists", "")),
            f"Selected job was not complete: {(job['experiment_id'], job['seed'])}")
    require(status.get("output_dir") == job.get("output_dir"),
            "Status output directory differs from its selected job row.")
    run = (repo / job["output_dir"]).resolve()
    checkpoint = run / f"{job['CHECKPOINT_PREFIX']}_full.pt"
    gate_path = run / "pretraining_behavior_gate.csv"
    require(checkpoint.is_file() and gate_path.is_file(),
            f"Selected run artifacts are missing: {run}")
    _, gate = read_csv(gate_path)
    require(bool(gate), f"Behavior gate is empty: {gate_path}")
    for row in gate:
        try:
            finite = math.isfinite(float(row.get("value", "nan")))
        except ValueError:
            finite = False
        require(finite, f"Non-finite behavior diagnostic: {gate_path}")
    if strict_gate_required:
        require(all(truth(row.get("pass", "")) for row in gate),
                f"A carried v2 job did not pass its strict gate: {gate_path}")


def equivalent_scientific_settings(old: dict[str, str],
                                   new: dict[str, str]) -> None:
    key = old["experiment_id"], old["seed"]
    common = (set(old) & set(new)) - IGNORED_JOB_FIELDS
    mismatches = {field: [old.get(field), new.get(field)] for field in common
                  if field != "PRETRAIN_BEHAVIOR_GATE_MODE" and
                  old.get(field) != new.get(field)}
    require(not mismatches, f"Scientific settings changed for {key}: {mismatches}")
    for field in ENV_FIELDS:
        if field == "PRETRAIN_BEHAVIOR_GATE_MODE":
            continue
        require(old.get(field) == new.get(field),
                f"Runtime setting {field} changed for {key}.")
    require(old.get("PRETRAIN_BEHAVIOR_GATE_MODE", "strict") == "strict",
            f"Original job did not use the strict legacy gate: {key}")
    require(new.get("PRETRAIN_BEHAVIOR_GATE_MODE") == "report_only",
            f"Retry job does not use the disclosed report-only gate: {key}")


def merge(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    for path in (args.output_jobs, args.output_status, args.output_manifest):
        require(not path.exists(), f"Operational merge output already exists: {path}")
    _, old_jobs_rows = read_csv(args.original_jobs)
    _, new_jobs_rows = read_csv(args.retry_jobs)
    old_status_fields, old_status_rows = read_csv(args.original_status)
    new_status_fields, new_status_rows = read_csv(args.retry_status)
    old_jobs, new_jobs = keyed(old_jobs_rows, "Original jobs"), keyed(new_jobs_rows, "Retry jobs")
    old_status = keyed(old_status_rows, "Original status")
    new_status = keyed(new_status_rows, "Retry status")
    require(len(old_jobs) == len(new_jobs) == len(old_status) == 130,
            "Original and retry job contracts must each contain exactly 130 keys.")
    require(set(old_jobs) == set(new_jobs) == set(old_status),
            "Original and retry job key sets differ.")
    old_passed = {key for key, row in old_status.items() if truth(row.get("passed", ""))}
    old_failed = set(old_status) - old_passed
    require(len(old_passed) == EXPECTED_ORIGINAL_PASSED and
            len(old_failed) == EXPECTED_RETRIED,
            f"Expected 70 v2 passes and 60 failures; found {len(old_passed)}/{len(old_failed)}.")
    require(set(new_status) == old_failed and len(new_status) == EXPECTED_RETRIED,
            "Retry status must contain exactly the original 60 failed keys.")
    require(all(truth(row.get("passed", "")) for row in new_status.values()),
            "At least one of the 60 v3 retry jobs failed.")
    require(old_status_fields == new_status_fields,
            "Original and retry status schemas differ.")

    for key in sorted(old_jobs):
        equivalent_scientific_settings(old_jobs[key], new_jobs[key])

    combined_jobs: list[dict[str, str]] = []
    combined_status: list[dict[str, str]] = []
    for key in sorted(old_jobs):
        if key in old_passed:
            job, status, source = dict(old_jobs[key]), dict(old_status[key]), "v2_carried"
            job["PRETRAIN_BEHAVIOR_GATE_MODE"] = "strict"
            verify_completed_job(repo, job, status, strict_gate_required=True)
        else:
            job, status, source = dict(new_jobs[key]), dict(new_status[key]), "v3_retry"
            verify_completed_job(repo, job, status, strict_gate_required=False)
        job["operational_source"] = source
        status["operational_source"] = source
        combined_jobs.append(job); combined_status.append(status)

    releases = {
        "original": verify_release(args.original_release, args.original_jobs),
        "retry": verify_release(args.retry_release, args.retry_jobs),
    }
    for parent in {args.output_jobs.parent, args.output_status.parent,
                   args.output_manifest.parent}:
        parent.mkdir(parents=True, exist_ok=True)
    job_fields = list(dict.fromkeys(
        field for row in combined_jobs for field in row))
    status_fields = list(dict.fromkeys(
        field for row in combined_status for field in row))

    def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.",
                                                       dir=path.parent)
        os.close(descriptor); temporary = Path(temporary_name)
        try:
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields,
                                        extrasaction="raise", restval="")
                writer.writeheader(); writer.writerows(rows)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    atomic_csv(args.output_jobs, job_fields, combined_jobs)
    atomic_csv(args.output_status, status_fields, combined_status)
    manifest = {
        "schema_version": 1,
        "status": "complete_70_v2_plus_60_v3_operational_merge",
        "job_count": 130,
        "v2_carried_count": len(old_passed),
        "v3_retry_count": len(old_failed),
        "seed_substitution_permitted": False,
        "causal_evaluation_accessed": False,
        "allowed_revision": "failed-path implementation repair plus intent-to-train gate reporting",
        "combined_jobs_sha256": sha256(args.output_jobs),
        "combined_status_sha256": sha256(args.output_status),
        "original_jobs_sha256": sha256(args.original_jobs),
        "original_status_sha256": sha256(args.original_status),
        "retry_jobs_sha256": sha256(args.retry_jobs),
        "retry_status_sha256": sha256(args.retry_status),
        "releases": releases,
        "claim_limit": "post-holdout explanatory mixed-revision evidence; not confirmatory",
    }
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_jobs.with_suffix(args.output_jobs.suffix + ".sha256").write_text(
        f"{manifest['combined_jobs_sha256']}  {args.output_jobs.name}\n", encoding="ascii")
    args.output_status.with_suffix(args.output_status.suffix + ".sha256").write_text(
        f"{manifest['combined_status_sha256']}  {args.output_status.name}\n", encoding="ascii")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--original-jobs", required=True, type=Path)
    parser.add_argument("--original-status", required=True, type=Path)
    parser.add_argument("--original-release", required=True, type=Path)
    parser.add_argument("--retry-jobs", required=True, type=Path)
    parser.add_argument("--retry-status", required=True, type=Path)
    parser.add_argument("--retry-release", required=True, type=Path)
    parser.add_argument("--output-jobs", required=True, type=Path)
    parser.add_argument("--output-status", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()
    for name in ("original_jobs", "original_status", "original_release",
                 "retry_jobs", "retry_status", "retry_release", "output_jobs",
                 "output_status", "output_manifest"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        result = merge(args)
    except (OperationalMergeError, ExtensionReleaseError, OSError, ValueError, KeyError,
            json.JSONDecodeError) as error:
        print(f"OPERATIONAL MERGE FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
