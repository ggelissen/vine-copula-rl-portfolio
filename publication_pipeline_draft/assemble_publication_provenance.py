#!/usr/bin/env python3
"""Assemble a self-contained audit pack without modifying locked OOS artifacts.

The pack records and verifies the successful locked batch, the pre-OOS training
and evaluation releases, every failed locked retry supplied by the operator,
raw-data provenance, and software-environment manifests.  Large training/data
objects are hash-inventoried by default and copied only when explicitly asked.

This command is a packaging/audit operation.  It never reruns training,
benchmark optimisation, policy inference, or performance scoring.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

try:  # package import in tests; script import in production CLI
    from .publication_pipeline import ProtocolError
except ImportError:  # pragma: no cover - direct CLI invocation
    from publication_pipeline import ProtocolError


SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_MAX_COPY_BYTES = 10 * 1024 * 1024
LARGE_ARTIFACT_SUFFIXES = {".pt", ".qs", ".rdata", ".tar", ".gz", ".zip"}


@dataclass
class ArtifactRecord:
    category: str
    logical_name: str
    source_path: str
    release_path: str
    size_bytes: int
    sha256: str
    copied: bool
    external: bool = False
    uri: str = ""
    license: str = ""
    omission_reason: str = ""


@dataclass
class HashRecord:
    hash_role: str
    logical_name: str
    algorithm: str
    digest: str
    source: str
    note: str = ""


@dataclass
class CheckRecord:
    check: str
    status: str
    detail: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ProtocolError(f"Required {label} is missing: {resolved}")
    return resolved


def require_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ProtocolError(f"Required {label} is missing: {resolved}")
    return resolved


def load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(require_file(path, label).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected a JSON object for {label}: {path}")
    return value


def parse_sidecar(sidecar: Path, archive: Path) -> str:
    text = require_file(sidecar, "archive SHA-256 sidecar").read_text(
        encoding="utf-8"
    ).strip()
    fields = text.split()
    if not fields or not SHA256_RE.fullmatch(fields[0].lower()):
        raise ProtocolError(f"Malformed SHA-256 sidecar: {sidecar}")
    expected = fields[0].lower()
    actual = sha256_file(require_file(archive, "archive"))
    if actual != expected:
        raise ProtocolError(
            f"Archive hash mismatch for {archive}: expected {expected}, found {actual}."
        )
    return actual


def safe_relative(value: str, label: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ProtocolError(f"Unsafe {label} path: {value}")
    return path


def parse_contents_manifest(root: Path) -> dict[str, str]:
    contents = require_file(root / "CONTENTS.sha256", "release CONTENTS.sha256")
    declared: dict[str, str] = {}
    for number, line in enumerate(contents.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if "  " not in line:
            raise ProtocolError(f"Malformed CONTENTS.sha256 line {number}: {line}")
        digest, relative = line.split("  ", 1)
        digest = digest.lower()
        if not SHA256_RE.fullmatch(digest):
            raise ProtocolError(f"Invalid SHA-256 on CONTENTS line {number}.")
        pure = safe_relative(relative, "CONTENTS")
        key = pure.as_posix()
        if key in declared:
            raise ProtocolError(f"Duplicate CONTENTS entry: {key}")
        artifact = root.joinpath(*pure.parts)
        if not artifact.is_file():
            raise ProtocolError(f"Release artifact declared but missing: {artifact}")
        actual = sha256_file(artifact)
        if actual != digest:
            raise ProtocolError(
                f"Release checksum mismatch for {artifact}: {actual} != {digest}."
            )
        declared[key] = digest
    if not declared:
        raise ProtocolError(f"Empty release checksum inventory: {contents}")
    return declared


def tar_file_entries(archive: Path) -> tuple[str, dict[str, tuple[tarfile.TarInfo, str]]]:
    files: dict[str, tuple[tarfile.TarInfo, str]] = {}
    roots: set[str] = set()
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            raw = member.name.replace("\\", "/").lstrip("./")
            pure = safe_relative(raw, "archive member")
            roots.add(pure.parts[0])
            if member.issym() or member.islnk():
                raise ProtocolError(f"Archive links are forbidden: {member.name}")
            if not member.isfile():
                continue
            stream = handle.extractfile(member)
            if stream is None:
                raise ProtocolError(f"Could not read archive member: {member.name}")
            relative = PurePosixPath(*pure.parts[1:]).as_posix()
            if not relative or relative in files:
                raise ProtocolError(f"Duplicate/invalid archive file: {member.name}")
            files[relative] = (member, sha256_bytes(stream.read()))
    if len(roots) != 1 or not files:
        raise ProtocolError(
            f"Archive must contain one top-level directory and files: {archive}"
        )
    return next(iter(roots)), files


def verify_archive_matches_directory(archive: Path, directory: Path) -> int:
    _, archived = tar_file_entries(archive)
    local = {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in directory.rglob("*")
        if path.is_file()
    }
    if set(archived) != set(local):
        raise ProtocolError(
            "Successful batch archive/directory file sets differ: "
            f"archive_only={sorted(set(archived) - set(local))}, "
            f"directory_only={sorted(set(local) - set(archived))}."
        )
    mismatches = [name for name in local if local[name] != archived[name][1]]
    if mismatches:
        raise ProtocolError(
            f"Successful batch archive/directory hashes differ: {mismatches}"
        )
    return len(local)


def read_archive_file(archive: Path, suffix: str) -> bytes:
    with tarfile.open(archive, "r:*") as handle:
        matches = []
        for member in handle.getmembers():
            raw = member.name.replace("\\", "/").lstrip("./")
            safe_relative(raw, "archive member")
            if member.isfile() and (raw == suffix or raw.endswith("/" + suffix)):
                matches.append(member)
        if len(matches) != 1:
            raise ProtocolError(
                f"Expected exactly one {suffix} in {archive}; found {len(matches)}."
            )
        stream = handle.extractfile(matches[0])
        if stream is None:
            raise ProtocolError(f"Could not read {matches[0].name} from {archive}.")
        return stream.read()


def read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    with require_file(path, label).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_file(
    source: Path,
    destination: Path,
    release_root: Path,
    category: str,
    logical_name: str,
    records: list[ArtifactRecord],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    records.append(
        ArtifactRecord(
            category=category,
            logical_name=logical_name,
            source_path=str(source),
            release_path=destination.relative_to(release_root).as_posix(),
            size_bytes=source.stat().st_size,
            sha256=sha256_file(source),
            copied=True,
        )
    )


def copy_release_tree(
    source: Path,
    destination: Path,
    release_root: Path,
    category: str,
    records: list[ArtifactRecord],
    copy_large: bool,
    max_copy_bytes: int,
) -> int:
    copied = 0
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source)
        large_suffix = any(
            path.name.lower().endswith(suffix) for suffix in LARGE_ARTIFACT_SUFFIXES
        )
        is_large = path.stat().st_size > max_copy_bytes or large_suffix
        target = destination / relative
        if copy_large or not is_large:
            copy_file(
                path, target, release_root, category, relative.as_posix(), records
            )
            copied += 1
        else:
            records.append(
                ArtifactRecord(
                    category=category,
                    logical_name=relative.as_posix(),
                    source_path=str(path),
                    release_path="",
                    size_bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                    copied=False,
                    omission_reason=(
                        "hash-inventoried large artifact; use --copy-large-artifacts"
                    ),
                )
            )
    return copied


def validate_release(
    root: Path,
    manifest_name: str,
    expected_status: str,
    checks: list[CheckRecord],
) -> tuple[dict[str, Any], dict[str, str]]:
    contents = parse_contents_manifest(root)
    manifest = load_json(root / manifest_name, manifest_name)
    if manifest.get("release_status") != expected_status:
        raise ProtocolError(
            f"{manifest_name} has release_status={manifest.get('release_status')!r}; "
            f"expected {expected_status!r}."
        )
    if bool(manifest.get("holdout_accessed_by_freezer", True)):
        raise ProtocolError(f"{manifest_name} was not frozen before holdout access.")
    checks.append(
        CheckRecord(
            check=f"{expected_status}_contents",
            status="pass",
            detail=f"Verified {len(contents)} release file hashes.",
        )
    )
    return manifest, contents


def validate_evaluation_aggregate(
    evaluation_release: Path,
    manifest: dict[str, Any],
    checks: list[CheckRecord],
    hashes: list[HashRecord],
) -> list[dict[str, str]]:
    inventory = read_csv_rows(
        evaluation_release / "evaluation_source_inventory.csv",
        "evaluation source inventory",
    )
    required = {"path", "sha256"}
    if not inventory or not required.issubset(inventory[0]):
        raise ProtocolError("Evaluation source inventory is empty or malformed.")
    payload = "\n".join(
        f"{row['sha256'].lower()}  {row['path']}" for row in inventory
    ).encode("utf-8")
    actual = sha256_bytes(payload)
    declared = str(manifest.get("evaluation_code_contract_sha256", "")).lower()
    if actual != declared:
        raise ProtocolError(
            f"Evaluation code/contract aggregate mismatch: {actual} != {declared}."
        )
    hashes.append(
        HashRecord(
            "evaluation_release_aggregate",
            "evaluation_code_contract",
            "sha256",
            actual,
            "evaluation_release_manifest.json",
            "Aggregate of frozen evaluation source and contract hashes, not a training hash.",
        )
    )
    for row in inventory:
        relative = row["path"].replace("\\", "/")
        if relative.endswith("evaluation_contract.json"):
            role = "evaluation_config"
        elif relative.endswith("benchmark_contract.json"):
            role = "benchmark_config"
        elif relative.endswith("config/config.yaml"):
            role = "runtime_config"
        else:
            role = "evaluation_code"
        digest = row["sha256"].lower()
        if not SHA256_RE.fullmatch(digest):
            raise ProtocolError(f"Invalid evaluation source hash for {relative}.")
        hashes.append(HashRecord(role, relative, "sha256", digest, relative))
    checks.append(
        CheckRecord(
            "evaluation_code_contract_aggregate",
            "pass",
            f"Aggregate {actual} matches {len(inventory)} frozen source/config rows.",
        )
    )
    return inventory


def validate_checkpoint_linkage(
    evaluation_release: Path,
    training_contents: dict[str, str],
    batch_directory: Path,
    checks: list[CheckRecord],
    hashes: list[HashRecord],
) -> None:
    declaration = read_csv_rows(
        evaluation_release / "strategy_declaration.csv", "strategy declaration"
    )
    declared = {
        row.get("checkpoint_sha256", "").lower()
        for row in declaration
        if SHA256_RE.fullmatch(row.get("checkpoint_sha256", "").lower())
    }
    training_full = {
        digest
        for relative, digest in training_contents.items()
        if relative.endswith("/td3_lstm_vine_full.pt")
    }
    batch_rows = read_csv_rows(batch_directory / "strategy_manifest.csv", "strategy manifest")
    batch_declared = {
        row.get("checkpoint_sha256", "").lower()
        for row in batch_rows
        if SHA256_RE.fullmatch(row.get("checkpoint_sha256", "").lower())
    }
    if not declared or declared != training_full or batch_declared != declared:
        raise ProtocolError(
            "Checkpoint hashes differ among training release, evaluation declaration, "
            "and successful batch manifest."
        )
    for index, digest in enumerate(sorted(declared), 1):
        hashes.append(
            HashRecord(
                "training_checkpoint",
                f"full_policy_checkpoint_{index:02d}",
                "sha256",
                digest,
                "training release / evaluation declaration / batch manifest",
            )
        )
    checks.append(
        CheckRecord(
            "checkpoint_release_linkage",
            "pass",
            f"Matched {len(declared)} full-policy checkpoints across all three layers.",
        )
    )


def add_training_hash_roles(
    training_release: Path,
    hashes: list[HashRecord],
) -> None:
    path = training_release / "training_snapshot_inventory.csv"
    rows = read_csv_rows(path, "training snapshot inventory")
    for row in rows:
        kind = row.get("artifact_kind", "").lower()
        role = "training_code" if kind == "code" else "training_data"
        digest = row.get("sha256", "").lower()
        algorithm = "sha256"
        if not SHA256_RE.fullmatch(digest):
            digest = row.get("expected_md5", "").lower()
            algorithm = "md5"
        if not digest:
            raise ProtocolError(
                f"Training snapshot row lacks a usable hash: {row.get('normalized_path')}"
            )
        hashes.append(
            HashRecord(
                role,
                row.get("normalized_path", ""),
                algorithm,
                digest,
                "training_snapshot_inventory.csv",
                "Training hashes are distinct from evaluation release/config hashes.",
            )
        )


def archive_incident(
    sequence: int,
    archive: Path,
    sidecar: Path,
    destination: Path,
    release_root: Path,
    records: list[ArtifactRecord],
    hashes: list[HashRecord],
) -> dict[str, Any]:
    archive_hash = parse_sidecar(sidecar, archive)
    manifest_bytes = read_archive_file(archive, "locked_batch_manifest.json")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "failed":
        raise ProtocolError(f"Retry archive is not a failed locked batch: {archive}")
    _, files = tar_file_entries(archive)
    incident_dir = destination / f"incident_{sequence:02d}"
    evidence = {
        name: read_archive_file(archive, name)
        for name in sorted(files)
        if name == "locked_batch_manifest.json" or name.startswith("command_logs/")
    }
    for relative, value in evidence.items():
        target = incident_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    copy_file(
        archive,
        incident_dir / archive.name,
        release_root,
        "failed_retry_archive",
        f"incident_{sequence:02d}",
        records,
    )
    copy_file(
        sidecar,
        incident_dir / sidecar.name,
        release_root,
        "failed_retry_sidecar",
        f"incident_{sequence:02d}_sidecar",
        records,
    )
    hashes.append(
        HashRecord(
            "failed_retry_archive",
            f"incident_{sequence:02d}",
            "sha256",
            archive_hash,
            archive.name,
        )
    )
    error = str(manifest.get("error", ""))
    match = re.search(r"Locked command ([^ ]+) failed", error)
    weight_count = sum(
        1 for name in files if "/weights" in "/" + name or name.startswith("weights/")
    )
    return {
        "sequence": sequence,
        "archive": archive.name,
        "sha256": archive_hash,
        "status": manifest.get("status", ""),
        "holdout_accessed": bool(manifest.get("holdout_accessed", False)),
        "failure_stage": match.group(1) if match else "unknown",
        "error_type": manifest.get("error_type", ""),
        "error": error,
        "contains_realized_panel": any(
            name.endswith("inputs/realized_asset_gross.csv") for name in files
        ),
        "weight_file_count": weight_count,
        "contains_publication_results": any(
            name.startswith("publication_results/") for name in files
        ),
        "log_file_count": sum(name.startswith("command_logs/") for name in files),
    }


def parse_local_raw_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ProtocolError("--raw-data must use LABEL=PATH syntax.")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise ProtocolError("--raw-data requires non-empty LABEL and PATH.")
    return label.strip(), Path(raw_path).resolve()


def parse_external_spec(value: str) -> tuple[str, str, str, str]:
    fields = value.split("|", 3)
    if len(fields) != 4 or any(not field.strip() for field in fields):
        raise ProtocolError(
            "--external-raw-data must use LABEL|SHA256|URI|LICENSE syntax."
        )
    label, digest, uri, license_name = (field.strip() for field in fields)
    if not SHA256_RE.fullmatch(digest.lower()):
        raise ProtocolError(f"External raw-data declaration has invalid SHA-256: {label}")
    return label, digest.lower(), uri, license_name


def write_contents(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "CONTENTS.sha256":
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "CONTENTS.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def deterministic_tar(source: Path, bundle: Path, root_name: str | None = None) -> None:
    name = root_name or source.name
    if not name or Path(name).name != name:
        raise ProtocolError(f"Archive root must be one stable path component: {name!r}")
    with bundle.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
            ) as handle:
                paths = [
                    source,
                    *sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()),
                ]
                for path in paths:
                    arcname = Path(name) / path.relative_to(source)
                    info = handle.gettarinfo(str(path), arcname=arcname.as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    if info.isdir():
                        info.mode = 0o755
                    elif info.isreg():
                        info.mode = 0o755 if info.mode & 0o111 else 0o644
                    if info.isreg():
                        with path.open("rb") as stream:
                            handle.addfile(info, stream)
                    else:
                        handle.addfile(info)


def assemble_publication_provenance(
    *,
    successful_batch_directory: Path,
    successful_batch_archive: Path,
    successful_batch_sidecar: Path,
    evaluation_release: Path,
    training_release: Path,
    failed_retries: list[tuple[Path, Path]],
    no_failed_retries: bool,
    raw_data: list[tuple[str, Path]],
    external_raw_data: list[tuple[str, str, str, str]],
    environment_manifests: list[Path],
    output: Path,
    bundle: Path,
    copy_raw_data: bool = False,
    copy_large_artifacts: bool = False,
    max_copy_bytes: int = DEFAULT_MAX_COPY_BYTES,
) -> dict[str, Any]:
    output = output.resolve()
    bundle = bundle.resolve()
    sidecar_output = bundle.with_suffix(bundle.suffix + ".sha256")
    if output.exists() or bundle.exists() or sidecar_output.exists():
        raise ProtocolError("Output, bundle, and sidecar paths must not already exist.")
    if max_copy_bytes <= 0:
        raise ProtocolError("max_copy_bytes must be positive.")
    if bool(failed_retries) == bool(no_failed_retries):
        raise ProtocolError(
            "Supply one or more --failed-retry pairs, or explicitly use --no-failed-retries."
        )
    if not raw_data and not external_raw_data:
        raise ProtocolError(
            "At least one local --raw-data or licensed --external-raw-data declaration is required."
        )
    if not environment_manifests:
        raise ProtocolError("At least one --environment-manifest is required.")

    batch_directory = require_directory(successful_batch_directory, "successful batch directory")
    batch_archive = require_file(successful_batch_archive, "successful batch archive")
    batch_sidecar = require_file(successful_batch_sidecar, "successful batch sidecar")
    evaluation_release = require_directory(evaluation_release, "evaluation release")
    training_release = require_directory(training_release, "training release")
    environment_manifests = [
        require_file(path, "environment manifest") for path in environment_manifests
    ]
    for label, path in raw_data:
        require_file(path, f"raw data {label}")

    checks: list[CheckRecord] = []
    records: list[ArtifactRecord] = []
    hashes: list[HashRecord] = []

    batch_hash = parse_sidecar(batch_sidecar, batch_archive)
    batch_file_count = verify_archive_matches_directory(batch_archive, batch_directory)
    batch_manifest = load_json(
        batch_directory / "locked_batch_manifest.json", "locked batch manifest"
    )
    if batch_manifest.get("status") != "complete" or not bool(
        batch_manifest.get("holdout_accessed", False)
    ):
        raise ProtocolError("Successful batch manifest is not complete/holdout-accessed.")
    checks.append(
        CheckRecord(
            "successful_batch_archive",
            "pass",
            f"Archive hash and all {batch_file_count} extracted files match.",
        )
    )
    hashes.append(
        HashRecord(
            "successful_locked_batch_archive",
            batch_archive.name,
            "sha256",
            batch_hash,
            str(batch_archive),
        )
    )

    evaluation_manifest, evaluation_contents = validate_release(
        evaluation_release,
        "evaluation_release_manifest.json",
        "frozen_pre_holdout_evaluation",
        checks,
    )
    training_manifest, training_contents = validate_release(
        training_release,
        "training_release_manifest.json",
        "frozen_pre_oos",
        checks,
    )
    validate_evaluation_aggregate(
        evaluation_release, evaluation_manifest, checks, hashes
    )
    expected_evaluation_hash = str(
        evaluation_manifest.get("evaluation_code_contract_sha256", "")
    ).lower()
    if str(batch_manifest.get("evaluation_release_sha256", "")).lower() != expected_evaluation_hash:
        raise ProtocolError(
            "Successful batch does not point to the supplied frozen evaluation release."
        )
    checks.append(
        CheckRecord(
            "batch_evaluation_release_link",
            "pass",
            f"Successful batch references evaluation aggregate {expected_evaluation_hash}.",
        )
    )
    validate_checkpoint_linkage(
        evaluation_release, training_contents, batch_directory, checks, hashes
    )
    add_training_hash_roles(training_release, hashes)
    hashes.append(
        HashRecord(
            "training_release_manifest",
            "training_release_manifest.json",
            "sha256",
            sha256_file(training_release / "training_release_manifest.json"),
            str(training_release),
        )
    )

    labels: set[str] = set()
    for label, _ in raw_data:
        if label in labels:
            raise ProtocolError(f"Duplicate raw-data label: {label}")
        labels.add(label)
    for label, _, _, _ in external_raw_data:
        if label in labels:
            raise ProtocolError(f"Duplicate raw-data label: {label}")
        labels.add(label)

    output.parent.mkdir(parents=True, exist_ok=True)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=output.parent))
    temporary_bundle_fd, temporary_bundle_name = tempfile.mkstemp(
        prefix=f".{bundle.name}_", suffix=".tmp", dir=bundle.parent
    )
    os.close(temporary_bundle_fd)
    temporary_bundle = Path(temporary_bundle_name)
    temporary_bundle.unlink()
    temporary_sidecar = temporary_bundle.with_suffix(temporary_bundle.suffix + ".sha256")
    try:
        copy_file(
            batch_archive,
            temporary / "successful_batch" / batch_archive.name,
            temporary,
            "successful_batch_archive",
            "successful_locked_batch",
            records,
        )
        copy_file(
            batch_sidecar,
            temporary / "successful_batch" / batch_sidecar.name,
            temporary,
            "successful_batch_sidecar",
            "successful_locked_batch_sidecar",
            records,
        )

        copied_evaluation = copy_release_tree(
            evaluation_release,
            temporary / "frozen_releases" / "evaluation",
            temporary,
            "evaluation_release",
            records,
            copy_large_artifacts,
            max_copy_bytes,
        )
        copied_training = copy_release_tree(
            training_release,
            temporary / "frozen_releases" / "training",
            temporary,
            "training_release",
            records,
            copy_large_artifacts,
            max_copy_bytes,
        )

        incidents = []
        for sequence, (archive, sidecar) in enumerate(failed_retries, 1):
            incidents.append(
                archive_incident(
                    sequence,
                    require_file(archive, "failed retry archive"),
                    require_file(sidecar, "failed retry sidecar"),
                    temporary / "failed_retries",
                    temporary,
                    records,
                    hashes,
                )
            )
        if no_failed_retries:
            checks.append(
                CheckRecord(
                    "incident_history",
                    "pass",
                    "Operator explicitly declared that no failed locked retries exist.",
                )
            )
        else:
            checks.append(
                CheckRecord(
                    "incident_history",
                    "warning",
                    f"Packaged {len(incidents)} failed holdout-access incidents; disclose them.",
                )
            )

        raw_dir = temporary / "raw_data"
        for label, path in raw_data:
            digest = sha256_file(path)
            hashes.append(
                HashRecord("raw_market_data", label, "sha256", digest, str(path))
            )
            if copy_raw_data:
                copy_file(
                    path,
                    raw_dir / f"{label}{path.suffix}",
                    temporary,
                    "raw_data",
                    label,
                    records,
                )
            else:
                records.append(
                    ArtifactRecord(
                        category="raw_data",
                        logical_name=label,
                        source_path=str(path),
                        release_path="",
                        size_bytes=path.stat().st_size,
                        sha256=digest,
                        copied=False,
                        omission_reason="hash-inventoried; use --copy-raw-data to copy",
                    )
                )
        for label, digest, uri, license_name in external_raw_data:
            hashes.append(
                HashRecord(
                    "external_licensed_raw_data",
                    label,
                    "sha256",
                    digest,
                    uri,
                    f"License: {license_name}",
                )
            )
            records.append(
                ArtifactRecord(
                    category="raw_data",
                    logical_name=label,
                    source_path="",
                    release_path="",
                    size_bytes=0,
                    sha256=digest,
                    copied=False,
                    external=True,
                    uri=uri,
                    license=license_name,
                    omission_reason="licensed/external artifact explicitly declared",
                )
            )
        checks.append(
            CheckRecord(
                "raw_data_provenance",
                "pass",
                f"Recorded {len(raw_data)} local and {len(external_raw_data)} external raw-data sources.",
            )
        )

        environment_names: set[str] = set()
        for path in environment_manifests:
            if path.name in environment_names:
                raise ProtocolError(f"Duplicate environment manifest basename: {path.name}")
            environment_names.add(path.name)
            copy_file(
                path,
                temporary / "environment" / path.name,
                temporary,
                "environment_manifest",
                path.name,
                records,
            )
            hashes.append(
                HashRecord(
                    "software_environment",
                    path.name,
                    "sha256",
                    sha256_file(path),
                    str(path),
                )
            )
        checks.append(
            CheckRecord(
                "software_environment",
                "pass",
                f"Copied {len(environment_manifests)} environment manifests.",
            )
        )

        omitted = sum(not item.copied and not item.external for item in records)
        if omitted:
            checks.append(
                CheckRecord(
                    "large_artifact_copy_policy",
                    "warning",
                    f"{omitted} existing artifacts are hash-inventoried but not copied.",
                )
            )
        else:
            checks.append(
                CheckRecord(
                    "large_artifact_copy_policy",
                    "pass",
                    "All local artifacts in the package inventory were copied.",
                )
            )
        checks.append(
            CheckRecord(
                "legacy_hash_semantics",
                "warning",
                "Legacy strategy config_sha256/code_sha256 may contain the evaluation aggregate; "
                "hash_registry.csv provides distinct semantic roles.",
            )
        )

        write_csv(
            temporary / "inventory.csv",
            (asdict(record) for record in records),
            list(ArtifactRecord.__dataclass_fields__),
        )
        write_csv(
            temporary / "hash_registry.csv",
            (asdict(record) for record in hashes),
            list(HashRecord.__dataclass_fields__),
        )
        write_csv(
            temporary / "incident_timeline.csv",
            incidents,
            [
                "sequence", "archive", "sha256", "status", "holdout_accessed",
                "failure_stage", "error_type", "error", "contains_realized_panel",
                "weight_file_count", "contains_publication_results", "log_file_count",
            ],
        )
        warning_count = sum(item.status == "warning" for item in checks)
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_with_warnings" if warning_count else "pass",
            "required_check_failures": 0,
            "warning_count": warning_count,
            "checks": [asdict(item) for item in checks],
        }
        (temporary / "validation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "release_status": "publication_provenance_evidence",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "successful_batch_archive": batch_archive.name,
            "successful_batch_archive_sha256": batch_hash,
            "successful_batch_file_count": batch_file_count,
            "evaluation_release_code_contract_sha256": expected_evaluation_hash,
            "evaluation_release_manifest_sha256": sha256_file(
                evaluation_release / "evaluation_release_manifest.json"
            ),
            "training_release_manifest_sha256": sha256_file(
                training_release / "training_release_manifest.json"
            ),
            "failed_retry_count": len(incidents),
            "raw_data_source_count": len(raw_data) + len(external_raw_data),
            "environment_manifest_count": len(environment_manifests),
            "copied_evaluation_release_files": copied_evaluation,
            "copied_training_release_files": copied_training,
            "copy_raw_data": copy_raw_data,
            "copy_large_artifacts": copy_large_artifacts,
            "max_copy_bytes": max_copy_bytes,
            "hash_semantics": {
                "evaluation_release_code_contract_sha256": (
                    "aggregate of frozen evaluation source and contracts"
                ),
                "training_code": "per-artifact training-source hash",
                "training_data": "per-artifact training-data hash",
                "checkpoint": "per-checkpoint SHA-256",
                "raw_market_data": "raw source-data SHA-256",
                "successful_locked_batch_archive": "immutable result archive SHA-256",
            },
            "scientific_note": (
                "This evidence package preserves and documents the existing locked result. "
                "It does not authorize replacement, retuning, or rescoring."
            ),
        }
        (temporary / "provenance_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        (temporary / "READ_ONLY_RELEASE.txt").write_text(
            "Supplementary provenance and incident evidence. Do not edit in place.\n"
            "The successful locked batch remains immutable and is copied byte-for-byte.\n",
            encoding="utf-8",
        )
        write_contents(temporary)
        parse_contents_manifest(temporary)

        deterministic_tar(temporary, temporary_bundle, root_name=output.name)
        bundle_hash = sha256_file(temporary_bundle)
        temporary_sidecar.write_text(
            f"{bundle_hash}  {bundle.name}\n", encoding="utf-8"
        )
        os.replace(temporary, output)
        os.replace(temporary_bundle, bundle)
        os.replace(temporary_sidecar, sidecar_output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        for path in [temporary_bundle, temporary_sidecar]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successful-batch-directory", required=True, type=Path)
    parser.add_argument("--successful-batch-archive", required=True, type=Path)
    parser.add_argument("--successful-batch-sidecar", required=True, type=Path)
    parser.add_argument("--evaluation-release", required=True, type=Path)
    parser.add_argument("--training-release", required=True, type=Path)
    parser.add_argument(
        "--failed-retry", action="append", nargs=2, metavar=("ARCHIVE", "SIDECAR"),
        default=[], help="Repeat once per failed locked retry, in chronological order."
    )
    parser.add_argument(
        "--no-failed-retries", action="store_true",
        help="Explicit declaration for experiments with no failed locked retry."
    )
    parser.add_argument(
        "--raw-data", action="append", default=[], metavar="LABEL=PATH",
        help="Existing raw source data. It is hashed but not copied by default."
    )
    parser.add_argument(
        "--external-raw-data", action="append", default=[],
        metavar="LABEL|SHA256|URI|LICENSE",
        help="Explicit licensed/external declaration when raw bytes cannot be copied."
    )
    parser.add_argument(
        "--environment-manifest", action="append", required=True, type=Path,
        help="Repeat for renv.lock, conda YAML, requirements lock, container digest, etc."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--copy-raw-data", action="store_true")
    parser.add_argument("--copy-large-artifacts", action="store_true")
    parser.add_argument("--max-copy-bytes", type=int, default=DEFAULT_MAX_COPY_BYTES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = assemble_publication_provenance(
        successful_batch_directory=args.successful_batch_directory,
        successful_batch_archive=args.successful_batch_archive,
        successful_batch_sidecar=args.successful_batch_sidecar,
        evaluation_release=args.evaluation_release,
        training_release=args.training_release,
        failed_retries=[(Path(pair[0]), Path(pair[1])) for pair in args.failed_retry],
        no_failed_retries=args.no_failed_retries,
        raw_data=[parse_local_raw_spec(value) for value in args.raw_data],
        external_raw_data=[parse_external_spec(value) for value in args.external_raw_data],
        environment_manifests=args.environment_manifest,
        output=args.output,
        bundle=args.bundle,
        copy_raw_data=args.copy_raw_data,
        copy_large_artifacts=args.copy_large_artifacts,
        max_copy_bytes=args.max_copy_bytes,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
