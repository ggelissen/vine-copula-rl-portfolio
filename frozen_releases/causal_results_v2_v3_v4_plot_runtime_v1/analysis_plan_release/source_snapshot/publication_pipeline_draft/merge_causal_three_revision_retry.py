#!/usr/bin/env python3
"""Merge 70 v2 successes, 31 v3 strict-path successes, and 29 v4 retries."""

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

from publication_pipeline_draft.behavior_gate_protocol import (
    BehaviorGateProtocolError,
    validate_report_only_trainer,
)
from publication_pipeline_draft.causal_ablation_protocol import ENV_FIELDS
from publication_pipeline_draft.extension_release import (
    ExtensionReleaseError,
    verify_frozen_extension_integrity,
)


class ThreeRevisionMergeError(RuntimeError):
    pass


V3_STRICT_ONLY_TRAINER_SHA256 = (
    "0dd04dae25d57fc84a48937555d9d29bed713ef8dac5b6443fb5581caa46a2fa")
V4_REPORT_ONLY_TRAINER_SHA256 = (
    "50f056bbfb7a2716eb0436223e3cb044d68544e1bdb744da722d7ceb9d7fd733")
COUNTS = {"v2": 70, "v3": 31, "v4": 29}
IGNORED_JOB_FIELDS = {"output_dir", "contract_sha256"}
UNCHANGED_TRAINING_SOURCES = (
    "run_with_config.r", "config/config.yaml", "helper/load_data.r",
    "helper/time_split.r", "helper/marginals.r", "helper/reproducibility.r",
    "benchmark_models/dynamic_vine_NN.r",
    "benchmark_models/expected_utility_single.r", "rl/rl_environment.r",
    "rl/action_projection.py", "rl/recurrent_baselines.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ThreeRevisionMergeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file(), f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def keyed(rows: list[dict[str, str]], label: str) -> dict[tuple[str, str], dict[str, str]]:
    result = {(row["experiment_id"], row["seed"]): row for row in rows}
    require(len(result) == len(rows), f"{label} contains duplicate keys.")
    return result


def release_evidence(release: Path, jobs: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest = verify_frozen_extension_integrity(release)
    require(manifest.get("causal_jobs_sha256") == sha256(jobs),
            f"Release is not bound to jobs: {release}")
    _, inventory_rows = read_csv(release / "source_inventory.csv")
    inventory = {row["path"]: row["sha256"] for row in inventory_rows}
    return ({"path": str(release.resolve()),
             "contents_sha256": manifest["release_contents_sha256"],
             "jobs_sha256": sha256(jobs)}, inventory)


def equivalent_settings(old: dict[str, str], new: dict[str, str],
                        allow_gate_revision: bool) -> None:
    key = old["experiment_id"], old["seed"]
    ignored = set(IGNORED_JOB_FIELDS)
    if allow_gate_revision:
        ignored.add("PRETRAIN_BEHAVIOR_GATE_MODE")
    mismatches = {field: [old.get(field), new.get(field)]
                  for field in (set(old) & set(new)) - ignored
                  if old.get(field) != new.get(field)}
    require(not mismatches, f"Scientific settings changed for {key}: {mismatches}")
    for field in ENV_FIELDS:
        if allow_gate_revision and field == "PRETRAIN_BEHAVIOR_GATE_MODE":
            continue
        require(old.get(field) == new.get(field),
                f"Runtime setting {field} changed for {key}.")


def validate_gate(path: Path, require_all_pass: bool) -> None:
    _, rows = read_csv(path)
    require(bool(rows), f"Behavior gate is empty: {path}")
    structural = {"gate_gross_mae", "max_position_limit_violation"}
    for row in rows:
        try:
            finite = math.isfinite(float(row.get("value", "nan")))
        except ValueError:
            finite = False
        require(finite, f"Non-finite behavior diagnostic: {path}")
        if row.get("metric") in structural:
            require(truth(row.get("pass", "")),
                    f"Hard-constraint behavior failure: {path}")
    if require_all_pass:
        require(all(truth(row.get("pass", "")) for row in rows),
                f"Carried strict-path job did not pass every gate: {path}")


def validate_complete(repo: Path, job: dict[str, str], status: dict[str, str],
                      require_all_gates: bool) -> None:
    require(status.get("exit_code") == "0" and truth(status.get("passed", "")) and
            truth(status.get("checkpoint_exists", "")) and
            truth(status.get("gate_exists", "")),
            f"Selected job is incomplete: {(job['experiment_id'], job['seed'])}")
    require(status.get("output_dir") == job.get("output_dir"),
            "Status and job output directories differ.")
    run = (repo / job["output_dir"]).resolve()
    checkpoint = run / f"{job['CHECKPOINT_PREFIX']}_full.pt"
    gate = run / "pretraining_behavior_gate.csv"
    require(checkpoint.is_file() and gate.is_file(), f"Run artifacts missing: {run}")
    validate_gate(gate, require_all_gates)


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor); temporary = Path(name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, restval="")
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def merge(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    for path in (args.output_jobs, args.output_status, args.output_manifest):
        require(not path.exists(), f"Merge output already exists: {path}")

    job_sets = []
    status_sets = []
    status_fields = []
    for job_path, status_path, label in (
            (args.v2_jobs, args.v2_status, "v2"),
            (args.v3_jobs, args.v3_status, "v3"),
            (args.v4_jobs, args.v4_status, "v4")):
        _, jobs = read_csv(job_path); fields, statuses = read_csv(status_path)
        job_sets.append(keyed(jobs, f"{label} jobs"))
        status_sets.append(keyed(statuses, f"{label} status"))
        status_fields.append(fields)
    v2_jobs, v3_jobs, v4_jobs = job_sets
    v2_status, v3_status, v4_status = status_sets
    require(len(v2_jobs) == len(v3_jobs) == len(v4_jobs) == 130,
            "Every revision must declare the same 130 job keys.")
    require(set(v2_jobs) == set(v3_jobs) == set(v4_jobs),
            "Revision job key sets differ.")
    require(status_fields[0] == status_fields[1] == status_fields[2],
            "Revision status schemas differ.")

    v2_pass = {key for key, row in v2_status.items() if truth(row["passed"])}
    v2_fail = set(v2_status) - v2_pass
    v3_pass = {key for key, row in v3_status.items() if truth(row["passed"])}
    v3_fail = set(v3_status) - v3_pass
    require(len(v2_pass) == COUNTS["v2"] and len(v2_fail) == 60,
            "V2 must contain exactly 70 passes and 60 failures.")
    require(set(v3_status) == v2_fail and len(v3_pass) == COUNTS["v3"] and
            len(v3_fail) == COUNTS["v4"],
            "V3 must cover the 60 v2 failures with an exact 31/29 split.")
    require(set(v4_status) == v3_fail and len(v4_status) == COUNTS["v4"] and
            all(truth(row["passed"]) for row in v4_status.values()),
            "V4 must complete the exact 29 v3 failures.")

    for key in sorted(v2_jobs):
        equivalent_settings(v2_jobs[key], v3_jobs[key], True)
        equivalent_settings(v3_jobs[key], v4_jobs[key], False)
    require(all(row.get("PRETRAIN_BEHAVIOR_GATE_MODE", "strict") == "strict"
                for row in v2_jobs.values()), "V2 was not the strict protocol.")
    require(all(row.get("PRETRAIN_BEHAVIOR_GATE_MODE") == "report_only"
                for row in v3_jobs.values()) and
            all(row.get("PRETRAIN_BEHAVIOR_GATE_MODE") == "report_only"
                for row in v4_jobs.values()),
            "V3/V4 job declarations must be report_only.")

    releases = {}
    inventories = {}
    for label, release, jobs in (
            ("v2", args.v2_release, args.v2_jobs),
            ("v3", args.v3_release, args.v3_jobs),
            ("v4", args.v4_release, args.v4_jobs)):
        releases[label], inventories[label] = release_evidence(release, jobs)
    require(inventories["v3"].get("rl/train_rl.r") ==
            V3_STRICT_ONLY_TRAINER_SHA256,
            "V3 is not the diagnosed strict-only trainer release.")
    require(inventories["v4"].get("rl/train_rl.r") ==
            V4_REPORT_ONLY_TRAINER_SHA256,
            "V4 is not the reviewed report-only trainer revision.")
    try:
        validate_report_only_trainer(
            args.v4_release / "source_snapshot/rl/train_rl.r")
    except BehaviorGateProtocolError as error:
        raise ThreeRevisionMergeError(str(error)) from error
    for relative in UNCHANGED_TRAINING_SOURCES:
        require(inventories["v3"].get(relative) == inventories["v4"].get(relative),
                f"Training source changed between v3 and v4: {relative}")

    combined_jobs = []
    combined_status = []
    for key in sorted(v2_jobs):
        if key in v2_pass:
            job, status, source = dict(v2_jobs[key]), dict(v2_status[key]), "v2_strict"
            validate_complete(repo, job, status, True)
            effective_gate = "strict"
        elif key in v3_pass:
            job, status, source = dict(v3_jobs[key]), dict(v3_status[key]), "v3_strict_path"
            validate_complete(repo, job, status, True)
            effective_gate = "strict"
        else:
            job, status, source = dict(v4_jobs[key]), dict(v4_status[key]), "v4_report_only"
            validate_complete(repo, job, status, False)
            effective_gate = "report_only"
        job["declared_pretrain_behavior_gate_mode"] = job.get(
            "PRETRAIN_BEHAVIOR_GATE_MODE", "strict")
        job["PRETRAIN_BEHAVIOR_GATE_MODE"] = effective_gate
        job["operational_source"] = source
        job["gate_path_equivalence"] = (
            "all_metrics_passed_no_branch_difference"
            if source == "v3_strict_path" else "not_applicable")
        status["operational_source"] = source
        combined_jobs.append(job); combined_status.append(status)

    job_fields = list(dict.fromkeys(field for row in combined_jobs for field in row))
    out_status_fields = list(dict.fromkeys(
        field for row in combined_status for field in row))
    atomic_csv(args.output_jobs, job_fields, combined_jobs)
    atomic_csv(args.output_status, out_status_fields, combined_status)
    manifest = {
        "schema_version": 1,
        "status": "complete_70_v2_plus_31_v3_plus_29_v4_operational_merge",
        "job_count": 130, "v2_carried_count": 70,
        "v3_carried_count": 31, "v4_retry_count": 29,
        "seed_substitution_permitted": False,
        "causal_evaluation_accessed": False,
        "v3_disposition": "strict-path successes retained only after all gates passed",
        "combined_jobs_sha256": sha256(args.output_jobs),
        "combined_status_sha256": sha256(args.output_status),
        "releases": releases,
        "claim_limit": "post-holdout explanatory three-revision evidence; not confirmatory",
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path, digest in ((args.output_jobs, manifest["combined_jobs_sha256"]),
                         (args.output_status, manifest["combined_status_sha256"])):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{digest}  {path.name}\n", encoding="ascii")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    for revision in ("v2", "v3", "v4"):
        parser.add_argument(f"--{revision}-jobs", required=True, type=Path)
        parser.add_argument(f"--{revision}-status", required=True, type=Path)
        parser.add_argument(f"--{revision}-release", required=True, type=Path)
    parser.add_argument("--output-jobs", required=True, type=Path)
    parser.add_argument("--output-status", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    try:
        result = merge(args)
    except (ThreeRevisionMergeError, ExtensionReleaseError, OSError, ValueError,
            KeyError, json.JSONDecodeError) as error:
        print(f"THREE-REVISION MERGE FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
