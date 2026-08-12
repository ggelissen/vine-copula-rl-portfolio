#!/usr/bin/env python3
"""Register an already frozen research archive without changing its contents.

The registrar is intentionally narrower than a freezer: it never re-runs an
experiment, edits an archive, or creates a new scientific result.  It verifies
the externally supplied checksum, the release's internal checksum inventory,
and the declared evidence class before copying the exact bytes into the local
``frozen_releases`` registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


class RegistrationError(RuntimeError):
    """Raised when an archive is not an admissible immutable release."""


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sidecar(path: Path, archive: Path) -> str:
    if not path.is_file():
        raise RegistrationError(f"Checksum sidecar not found: {path}")
    fields = path.read_text(encoding="utf-8").strip().split()
    if not fields or not SHA256_RE.fullmatch(fields[0].lower()):
        raise RegistrationError("Checksum sidecar does not begin with SHA-256.")
    if len(fields) > 1 and Path(fields[-1].lstrip("* ")).name != archive.name:
        raise RegistrationError("Checksum sidecar names a different archive.")
    return fields[0].lower()


def safe_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise RegistrationError(f"Unsafe archive member: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise RegistrationError(f"Links/devices are forbidden: {member.name}")
        normalized = path.as_posix().lstrip("./")
        if normalized in members:
            raise RegistrationError(f"Duplicate archive member: {normalized}")
        members[normalized] = member
    return members


def member_bytes(
    archive: tarfile.TarFile, members: dict[str, tarfile.TarInfo], name: str
) -> bytes:
    member = members.get(name)
    if member is None or not member.isfile():
        raise RegistrationError(f"Required file is missing from archive: {name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise RegistrationError(f"Could not read archive member: {name}")
    return stream.read()


def unique_suffix(members: dict[str, tarfile.TarInfo], suffix: str) -> str:
    matches = [name for name in members if name.endswith(suffix)]
    if len(matches) != 1:
        raise RegistrationError(
            f"Expected exactly one {suffix}; found {len(matches)}."
        )
    return matches[0]


def verify_internal_release(archive_path: Path) -> dict[str, Any]:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = safe_members(archive)
        manifest_name = unique_suffix(
            members, "post_holdout_explanatory_release_manifest.json"
        )
        root = str(PurePosixPath(manifest_name).parent)
        prefix = "" if root == "." else root.rstrip("/") + "/"
        read_only_name = prefix + "READ_ONLY_RELEASE.txt"
        contents_name = prefix + "CONTENTS.sha256"
        if not member_bytes(archive, members, read_only_name).strip():
            raise RegistrationError("READ_ONLY_RELEASE.txt is empty.")
        try:
            manifest = json.loads(
                member_bytes(archive, members, manifest_name).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RegistrationError("Release manifest is not valid UTF-8 JSON.") from error
        required = {
            "release_status": "frozen_post_holdout_explanatory_ablation",
            "evidence_class": "post_holdout_explanatory",
        }
        for field, expected in required.items():
            if manifest.get(field) != expected:
                raise RegistrationError(
                    f"Manifest {field}={manifest.get(field)!r}; expected {expected!r}."
                )
        if manifest.get("confirmatory_claims_permitted") not in (False, None):
            raise RegistrationError("A post-holdout release cannot permit confirmatory claims.")

        inventory = member_bytes(archive, members, contents_name).decode("utf-8")
        verified = 0
        for line_number, line in enumerate(inventory.splitlines(), start=1):
            if not line.strip():
                continue
            match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
            if match is None:
                raise RegistrationError(
                    f"Malformed CONTENTS.sha256 line {line_number}."
                )
            expected, relative = match.group(1).lower(), match.group(2)
            relative_path = PurePosixPath(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise RegistrationError(f"Unsafe checksum path: {relative}")
            member_name = prefix + relative_path.as_posix()
            actual = hashlib.sha256(
                member_bytes(archive, members, member_name)
            ).hexdigest()
            if actual != expected:
                raise RegistrationError(f"Internal checksum mismatch: {relative}")
            verified += 1
        if verified == 0:
            raise RegistrationError("CONTENTS.sha256 contains no files.")
        return {
            "release_manifest": manifest,
            "release_manifest_member": manifest_name,
            "release_manifest_sha256": hashlib.sha256(
                member_bytes(archive, members, manifest_name)
            ).hexdigest(),
            "verified_internal_files": verified,
        }


def register(
    archive: Path, checksum: Path, output: Path, *, registration_id: str
) -> dict[str, Any]:
    archive = archive.resolve()
    checksum = checksum.resolve()
    output = output.resolve()
    if not archive.is_file():
        raise RegistrationError(f"Frozen archive not found: {archive}")
    if output.exists():
        raise RegistrationError(f"Registry destination already exists: {output}")
    expected = read_sidecar(checksum, archive)
    actual = sha256_file(archive)
    if actual != expected:
        raise RegistrationError(
            f"Outer archive checksum mismatch: expected {expected}, got {actual}."
        )
    internal = verify_internal_release(archive)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        copied_archive = temporary / archive.name
        shutil.copyfile(archive, copied_archive)
        (temporary / f"{archive.name}.sha256").write_text(
            f"{actual}  {archive.name}\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "registration_id": registration_id,
            "registered_utc": datetime.now(timezone.utc).isoformat(),
            "operation": "byte_exact_registration_no_reexecution",
            "archive_name": archive.name,
            "archive_sha256": actual,
            "release_status": internal["release_manifest"]["release_status"],
            "evidence_class": internal["release_manifest"]["evidence_class"],
            "confirmatory_claims_permitted": False,
            "release_manifest_member": internal["release_manifest_member"],
            "release_manifest_sha256": internal["release_manifest_sha256"],
            "verified_internal_files": internal["verified_internal_files"],
            "scientific_note": (
                "The registered archive is a consumed-holdout explanatory release. "
                "Registration does not create new evidence or permit confirmatory claims."
            ),
        }
        (temporary / "registration_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "READ_ONLY_REGISTRATION.txt").write_text(
            "Do not edit this directory. Replace it only with a new versioned registration.\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--checksum", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registration-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = register(
            args.archive,
            args.checksum,
            args.output,
            registration_id=args.registration_id,
        )
    except RegistrationError as error:
        print(f"REGISTRATION FAILURE: {error}")
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
