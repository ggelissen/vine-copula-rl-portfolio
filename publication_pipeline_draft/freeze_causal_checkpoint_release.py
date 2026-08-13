#!/usr/bin/env python3
"""Consolidate the audited 70/31/29 causal checkpoints into one release.

The three operational revisions are scientifically one intent-to-train cohort.
This freezer resolves each audited checkpoint, verifies its recorded hash and
metadata cardinality, and creates a content-addressed canonical checkpoint
release.  Original run trees must not be removed until this release and its
archive have both passed checksum verification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any


class ReleaseError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Checkpoint audit not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(rows, "Checkpoint audit is empty.")
    return rows


def resolve_checkpoint(recorded: str, repo_root: Path,
                       remaps: list[tuple[str, Path]]) -> Path:
    normalized = recorded.replace("\\", "/")
    direct = Path(recorded)
    candidates: list[Path] = []
    if direct.is_absolute():
        candidates.append(direct)
    candidates.append(repo_root / direct)
    for prefix, replacement in remaps:
        if normalized == prefix or normalized.startswith(prefix.rstrip("/") + "/"):
            suffix = normalized[len(prefix.rstrip("/")):].lstrip("/")
            candidates.append(replacement / Path(suffix))
    existing: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved not in existing:
            existing.append(resolved)
    require(len(existing) == 1,
            f"Checkpoint path is missing or ambiguous ({len(existing)} matches): {recorded}")
    return existing[0]


def parse_remaps(values: list[str]) -> list[tuple[str, Path]]:
    remaps: list[tuple[str, Path]] = []
    for value in values:
        require("=" in value, "--path-remap must be RECORDED_PREFIX=LIVE_PREFIX")
        source, destination = value.split("=", 1)
        require(bool(source.strip()) and bool(destination.strip()),
                "--path-remap contains an empty prefix.")
        remaps.append((source.replace("\\", "/").rstrip("/"),
                       Path(destination).resolve()))
    return remaps


def safe_arcname(name: str) -> str:
    return name.replace("\\", "/")


def write_archive(source: Path, archive: Path) -> str:
    require(not archive.exists() and
            not archive.with_suffix(archive.suffix + ".sha256").exists(),
            f"Archive output already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    with tarfile.open(temporary, "w:gz", format=tarfile.PAX_FORMAT) as bundle:
        for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
            arcname = Path(source.name) / path.relative_to(source)
            bundle.add(path, arcname=safe_arcname(arcname.as_posix()),
                       recursive=False)
    os.replace(temporary, archive)
    digest = sha256(archive)
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return digest


def freeze(repo_root: Path, audit: Path, output: Path, archive: Path | None,
           remaps: list[tuple[str, Path]]) -> dict[str, Any]:
    require(not output.exists(), f"Release output already exists: {output}")
    rows = read_rows(audit)
    required = {
        "experiment_id", "seed", "checkpoint", "sha256", "size_bytes",
        "checkpoint_schema", "rl_algorithm", "policy_encoder",
        "vine_feature_mode", "cvar_observation_mode", "cvar_reward_mode",
        "pretrain_data_mode", "run_finetune", "all_tensors_finite",
        "behavior_gate_pass", "behavior_gate_mode", "operational_source",
    }
    missing = required - set(rows[0])
    require(not missing, f"Checkpoint audit is missing columns: {sorted(missing)}")
    keys = {(row["experiment_id"], int(row["seed"])) for row in rows}
    require(len(rows) == len(keys) == 130, "Expected exactly 130 unique audited policies.")
    experiments = {row["experiment_id"] for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    require(len(experiments) == 13 and len(seeds) == 10,
            "Expected thirteen experiments and ten matched seeds.")
    sources: dict[str, int] = {}
    for row in rows:
        sources[row["operational_source"]] = sources.get(row["operational_source"], 0) + 1
    require(sources == {"v2_strict": 70, "v3_strict_path": 31,
                        "v4_report_only": 29},
            f"Operational carry-forward differs from 70/31/29: {sources}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    inventory: list[dict[str, Any]] = []
    try:
        checkpoint_root = temporary / "checkpoints"
        for row in sorted(rows, key=lambda item: (item["experiment_id"],
                                                   int(item["seed"]))):
            # Economic behavior warnings were retained under the registered
            # intent-to-train/report-only policy. They are metadata, not a
            # reason to remove an audited finite checkpoint from this release.
            require(row["all_tensors_finite"].lower() == "true",
                    f"Non-finite checkpoint in audit: {row['experiment_id']} {row['seed']}")
            source = resolve_checkpoint(row["checkpoint"], repo_root, remaps)
            expected = row["sha256"].lower()
            require(len(expected) == 64 and sha256(source) == expected,
                    f"Checkpoint hash mismatch: {source}")
            require(source.stat().st_size == int(row["size_bytes"]),
                    f"Checkpoint size mismatch: {source}")
            relative = (Path("checkpoints") / row["experiment_id"] /
                        f"seed_{row['seed']}" / source.name)
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            require(sha256(destination) == expected,
                    f"Copied checkpoint hash mismatch: {destination}")
            inventory.append({
                "experiment_id": row["experiment_id"],
                "seed": int(row["seed"]),
                "checkpoint": relative.as_posix(),
                "sha256": expected,
                "size_bytes": int(row["size_bytes"]),
                "checkpoint_schema": row["checkpoint_schema"],
                "rl_algorithm": row["rl_algorithm"],
                "policy_encoder": row["policy_encoder"],
                "vine_feature_mode": row["vine_feature_mode"],
                "cvar_observation_mode": row["cvar_observation_mode"],
                "cvar_reward_mode": row["cvar_reward_mode"],
                "pretrain_data_mode": row["pretrain_data_mode"],
                "run_finetune": row["run_finetune"],
                "operational_source": row["operational_source"],
                "original_recorded_path": row["checkpoint"],
            })
        inventory_path = temporary / "causal_checkpoint_inventory.csv"
        with inventory_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
            writer.writeheader()
            writer.writerows(inventory)
        shutil.copy2(audit, temporary / "checkpoint_audit_snapshot.csv")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "release_status": "frozen_causal_checkpoint_release",
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "policy_count": 130,
            "experiment_count": 13,
            "seed_count": 10,
            "operational_source_counts": sources,
            "audit_sha256": sha256(audit),
            "inventory_sha256": sha256(inventory_path),
            "all_tensors_finite": True,
            "scientific_note": (
                "Canonical replay material for the audited 70 v2 plus 31 v3 plus "
                "29 v4 intent-to-train causal cohort. Operational revisions are "
                "preserved as metadata and do not authorize confirmatory claims."
            ),
            "deletion_authorization": (
                "Original causal run trees may be removed only after CONTENTS.sha256 "
                "and the optional archive sidecar both pass independently."
            ),
        }
        (temporary / "causal_checkpoint_release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "READ_ONLY_RELEASE.txt").write_text(
            "Immutable audited causal checkpoint release. Do not edit.\n",
            encoding="utf-8")
        files = sorted(
            (path for path in temporary.rglob("*") if path.is_file() and
             path.name != "CONTENTS.sha256"),
            key=lambda path: path.relative_to(temporary).as_posix(),
        )
        with (temporary / "CONTENTS.sha256").open(
                "w", encoding="ascii", newline="\n") as stream:
            for path in files:
                stream.write(
                    f"{sha256(path)}  {path.relative_to(temporary).as_posix()}\n")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    if archive is not None:
        manifest["archive_sha256"] = write_archive(output, archive)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--path-remap", action="append", default=[],
        help="Map an audited recorded prefix to a live prefix; repeatable.")
    arguments = parser.parse_args()
    try:
        result = freeze(
            arguments.repo_root.resolve(), arguments.audit.resolve(),
            arguments.output.resolve(),
            arguments.archive.resolve() if arguments.archive else None,
            parse_remaps(arguments.path_remap),
        )
    except (OSError, ValueError, ReleaseError) as error:
        print(f"CAUSAL CHECKPOINT RELEASE FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
