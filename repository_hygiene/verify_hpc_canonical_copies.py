#!/usr/bin/env python3
"""Fail-closed canonical-copy audit for the completed HPC research tree.

This script is intentionally read-only.  It verifies content manifests, archive
sidecars, and checkpoint copies, then writes an authorization report describing
which generated work trees may be removed.  It never removes or modifies the
scientific artifacts being checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import BinaryIO, Iterable


HASH_LINE = re.compile(r"^([0-9a-fA-F]{64})[ \t]+[* ]?(.*?)\s*$")
CHECKPOINT_FILES = ("td3_lstm_vine_pretrained.pt", "td3_lstm_vine_full.pt")


class AuditFailure(RuntimeError):
    """Raised whenever canonical evidence is absent or inconsistent."""


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def read_hash_manifest(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        raise AuditFailure(f"Missing checksum manifest: {path}")
    rows: list[tuple[str, str]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        match = HASH_LINE.match(raw)
        if not match or not match.group(2):
            raise AuditFailure(f"Invalid checksum line {path}:{number}: {raw!r}")
        rows.append((match.group(1).lower(), match.group(2)))
    if not rows:
        raise AuditFailure(f"Empty checksum manifest: {path}")
    return rows


def resolve_manifest_target(root: Path, manifest: Path, name: str) -> Path:
    candidate = Path(name)
    choices = [candidate] if candidate.is_absolute() else [manifest.parent / candidate, root / candidate]
    for choice in choices:
        resolved = choice.resolve()
        if inside(resolved, root) and resolved.is_file():
            return resolved
    raise AuditFailure(f"Checksum target not found for {manifest}: {name}")


def verify_manifest(root: Path, manifest: Path) -> dict[str, object]:
    rows = read_hash_manifest(manifest)
    checked: list[str] = []
    for expected, name in rows:
        target = resolve_manifest_target(root, manifest, name)
        actual = sha256_file(target)
        if actual != expected:
            raise AuditFailure(
                f"Checksum mismatch: {target}\nexpected={expected}\nactual={actual}"
            )
        checked.append(target.relative_to(root).as_posix())
    return {
        "manifest": manifest.relative_to(root).as_posix(),
        "file_count": len(checked),
        "manifest_sha256": sha256_file(manifest),
    }


def verify_release_manifests(root: Path) -> list[dict[str, object]]:
    search_roots = (
        root / "frozen_releases",
        root / "locked_evaluation",
        root / "secondary_evaluation",
        root / "manuscript_revision_causal_v1" / "publication_terminal_v1",
        root / "manuscript_revision_causal_v1" / "publication_mixed_pretraining_v1",
        root / "data" / "ablation_training_bundles",
    )
    manifests: list[Path] = []
    for directory in search_roots:
        if directory.exists():
            manifests.extend(directory.rglob("CONTENTS.sha256"))
    if not manifests:
        raise AuditFailure("No canonical CONTENTS.sha256 manifests were found.")
    results = []
    for index, manifest in enumerate(sorted(set(manifests)), 1):
        print(f"[contents {index}/{len(set(manifests))}] {manifest.relative_to(root)}", flush=True)
        results.append(verify_manifest(root, manifest))
    return results


def verify_archive_sidecars(root: Path) -> list[dict[str, object]]:
    sidecars = [
        path
        for path in root.rglob("*.tar.gz.sha256")
        if ".git" not in path.parts and "tmp" not in path.parts
    ]
    if not sidecars:
        raise AuditFailure("No tar.gz.sha256 archive sidecars were found.")
    results = []
    for index, sidecar in enumerate(sorted(sidecars), 1):
        print(f"[archive {index}/{len(sidecars)}] {sidecar.relative_to(root)}", flush=True)
        result = verify_manifest(root, sidecar)
        if result["file_count"] != 1:
            raise AuditFailure(f"Archive sidecar must identify exactly one file: {sidecar}")
        results.append(result)
    return results


def compare_checkpoint_directories(
    root: Path,
    raw_pattern: str,
    canonical_pattern: str,
    seeds: Iterable[int],
    label: str,
) -> dict[str, object]:
    compared = 0
    raw_seen = 0
    for seed in seeds:
        raw_dir = root / raw_pattern.format(seed=seed)
        canonical_dir = root / canonical_pattern.format(seed=seed)
        if raw_dir.exists():
            raw_seen += 1
        if not canonical_dir.is_dir():
            raise AuditFailure(f"Missing canonical checkpoint directory: {canonical_dir}")
        for filename in CHECKPOINT_FILES:
            canonical = canonical_dir / filename
            if not canonical.is_file():
                raise AuditFailure(f"Missing canonical checkpoint: {canonical}")
            if raw_dir.exists():
                raw = raw_dir / filename
                if not raw.is_file():
                    raise AuditFailure(f"Raw run is missing checkpoint: {raw}")
                raw_hash = sha256_file(raw)
                canonical_hash = sha256_file(canonical)
                if raw_hash != canonical_hash:
                    raise AuditFailure(
                        f"Checkpoint mismatch ({label}, seed {seed}, {filename}):\n"
                        f"raw={raw_hash}\ncanonical={canonical_hash}"
                    )
                compared += 1
    return {
        "label": label,
        "seed_count": len(tuple(seeds)),
        "raw_seed_directories_present": raw_seen,
        "checkpoint_files_compared": compared,
        "canonical_checkpoint_files_required": len(tuple(seeds)) * len(CHECKPOINT_FILES),
        "status": "matched" if raw_seen else "canonical_present_raw_already_absent",
    }


def tar_member_by_suffix(archive: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    normalized = suffix.replace(os.sep, "/").lstrip("/")
    matches = [
        member
        for member in archive.getmembers()
        if member.isfile()
        and (member.name == normalized or member.name.endswith("/" + normalized))
    ]
    if len(matches) != 1:
        raise AuditFailure(
            f"Expected exactly one archive member ending in {normalized!r}; found {len(matches)}"
        )
    return matches[0]


def compare_checkpoint_archive(
    root: Path,
    raw_root: str,
    archive_path: str,
    experiment_checkpoints: dict[str, tuple[str, ...]],
    seeds: Iterable[int],
    label: str,
) -> dict[str, object]:
    raw_base = root / raw_root
    archive_file = root / archive_path
    if not archive_file.is_file():
        raise AuditFailure(f"Missing checkpoint archive: {archive_file}")
    compared = 0
    required = 0
    with tarfile.open(archive_file, "r:gz") as archive:
        for experiment, checkpoint_files in experiment_checkpoints.items():
            for seed in seeds:
                for filename in checkpoint_files:
                    required += 1
                    suffix = f"{raw_root}/{experiment}/seed_{seed}/{filename}"
                    member = tar_member_by_suffix(archive, suffix)
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise AuditFailure(f"Could not read checkpoint member: {member.name}")
                    archived_hash = sha256_stream(extracted)
                    raw = raw_base / experiment / f"seed_{seed}" / filename
                    if raw_base.exists():
                        if not raw.is_file():
                            raise AuditFailure(f"Raw run is missing checkpoint: {raw}")
                        raw_hash = sha256_file(raw)
                        if raw_hash != archived_hash:
                            raise AuditFailure(
                                f"Checkpoint archive mismatch ({label}): {raw}\n"
                                f"raw={raw_hash}\narchive={archived_hash}"
                            )
                        compared += 1
    return {
        "label": label,
        "archive": archive_path,
        "experiment_count": len(experiment_checkpoints),
        "seed_count": len(tuple(seeds)),
        "canonical_checkpoint_files_required": required,
        "checkpoint_files_compared": compared,
        "status": "matched" if raw_base.exists() else "canonical_present_raw_already_absent",
    }


def require_paths(root: Path, paths: Iterable[str]) -> list[str]:
    checked: list[str] = []
    for relative in paths:
        path = root / relative
        if not path.exists():
            raise AuditFailure(f"Required canonical artifact is missing: {path}")
        checked.append(relative)
    return checked


def authorization_text() -> str:
    return """CANONICAL COPY AUDIT PASSED
===========================

The verifier found no missing or mismatched canonical evidence. The following
conditional cleanup groups from HPC_SAFE_DELETION_AUDIT_2026-08-18.txt are now
authorized:

1. Full-model duplicated training work trees
   data/rl_runs/
   data/publication_training_artifacts_20seeds/

2. No-vine duplicated/superseded training work trees
   data/no_vine_rl_runs_secondary_v3/
   data/publication_no_vine_training_artifacts_10seeds/
   data/no_vine_rl_runs_4gpu/

3. Masked-pretraining duplicated work products
   data/masked_pretraining_control_runs_v1/
   analysis_outputs/masked_pretraining_controls_v1_weights/
   analysis_outputs/masked_pretraining_controls_v1_audit/
   logs/masked_pretraining_controls_v1/
   logs/masked_pretraining_controls_v1*.log

4. Mixed-pretraining duplicated work products
   data/mixed_pretraining_runs_v1/
   analysis_outputs/mixed_pretraining_response_v1_weights/
   analysis_outputs/mixed_pretraining_response_v1_audit/
   logs/mixed_pretraining_response_v1/
   logs/mixed_pretraining_response_v1*.log

5. Causal replay/accounting intermediates listed in Section C5 of the audit.

6. The extracted secondary_evaluation directory listed in Section C6, provided
   no current command uses it as an input.

7. The terminal clean-room duplicate listed in Section C7.

Also delete every Section A path; those did not require canonical-copy checks.

STILL BLOCKED — DO NOT DELETE
=============================

data/focused_original_7asset_runs_v1/
data/publication_extension_runs_v4/
data/synthetic_dose_response_runs_v1/
data/synthetic_presentation_response_runs_v2/

The repository tree did not show complete independent checkpoint archives for
these four raw run roots. Their result archives do not prove checkpoint
preservation. Keep them until purpose-built checkpoint releases exist.

Retain every path in Section D2–D6 of the deletion audit, all frozen releases,
all archive sidecars, protocol/environment evidence, source, tests, and active
manuscript artifacts.
"""


def audit(repo_root: Path, output: Path) -> dict[str, object]:
    root = repo_root.resolve()
    if not (root / "publication_pipeline_draft").is_dir():
        raise AuditFailure(f"Not a repository root: {root}")

    required = require_paths(
        root,
        (
            "frozen_releases/training_schema5_v1",
            "frozen_releases/no_vine_schema5_secondary_v1",
            "frozen_releases/causal_results_v2_v3_v4_plot_runtime_v1",
            "frozen_releases/final_evidence/causal_results_v2_v3_v4_plot_runtime_v1.tar.gz",
            "frozen_releases/masked_pretraining_controls_v1",
            "frozen_releases/mixed_pretraining_response_v1_evidence_v1",
            "frozen_releases/terminal_robustness_v1",
            "locked_evaluation/main_oos_v4_operational_retry",
            "manuscript_revision_causal_v1/publication_terminal_v1",
            "manuscript_revision_causal_v1/publication_mixed_pretraining_v1",
        ),
    )

    contents = verify_release_manifests(root)
    archives = verify_archive_sidecars(root)

    checkpoint_checks = [
        compare_checkpoint_directories(
            root,
            "data/rl_runs/schema5_seed_{seed}",
            "frozen_releases/training_schema5_v1/seeds/seed_{seed}",
            range(20260741, 20260761),
            "full_schema5_20seed",
        ),
        compare_checkpoint_directories(
            root,
            "data/no_vine_rl_runs_secondary_v3/seed_{seed}",
            "frozen_releases/no_vine_schema5_secondary_v1/seeds/seed_{seed}",
            range(20260841, 20260851),
            "no_vine_10seed",
        ),
        compare_checkpoint_archive(
            root,
            "data/masked_pretraining_control_runs_v1",
            "masked_pretraining_controls_v1_checkpoints.tar.gz",
            {
                "masked_historical_prefix_1000_presentations": (
                    "td3_lstm_masked_histprefix1000_pretrained.pt",
                    "td3_lstm_masked_histprefix1000_full.pt",
                ),
                "masked_moving_block_bootstrap_1000_presentations": (
                    "td3_lstm_masked_mbb1000_pretrained.pt",
                    "td3_lstm_masked_mbb1000_full.pt",
                ),
            },
            range(20261001, 20261011),
            "masked_pretraining_controls",
        ),
        compare_checkpoint_archive(
            root,
            "data/mixed_pretraining_runs_v1",
            "mixed_pretraining_response_v1_checkpoints.tar.gz",
            {
                "mixed_100synthetic_61historical_pretrain_plus_historical_finetune": (
                    "td3_lstm_mixed100synth61hist1000_pretrained.pt",
                    "td3_lstm_mixed100synth61hist1000_full.pt",
                ),
            },
            range(20261001, 20261011),
            "mixed_pretraining",
        ),
    ]

    report = {
        "schema_version": 1,
        "status": "canonical_copy_audit_passed",
        "repo_root": root.as_posix(),
        "required_artifacts": required,
        "contents_manifest_count": len(contents),
        "archive_sidecar_count": len(archives),
        "contents_manifests": contents,
        "archive_sidecars": archives,
        "checkpoint_checks": checkpoint_checks,
        "authorized_cleanup_groups": ["A", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        "blocked_raw_run_roots": [
            "data/focused_original_7asset_runs_v1",
            "data/publication_extension_runs_v4",
            "data/synthetic_dose_response_runs_v1",
            "data/synthetic_presentation_response_runs_v2",
        ],
    }

    output_dir = output.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "canonical_copy_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "AUTHORIZED_CONDITIONAL_DELETIONS.txt").write_text(
        authorization_text(), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("repository_hygiene/hpc_canonical_copy_audit_v1"),
    )
    arguments = parser.parse_args()
    try:
        report = audit(arguments.repo_root, arguments.output)
    except (AuditFailure, OSError, tarfile.TarError) as error:
        print(f"CANONICAL COPY AUDIT FAILURE: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": report["status"],
        "contents_manifest_count": report["contents_manifest_count"],
        "archive_sidecar_count": report["archive_sidecar_count"],
        "checkpoint_checks": report["checkpoint_checks"],
        "output": arguments.output.as_posix(),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
