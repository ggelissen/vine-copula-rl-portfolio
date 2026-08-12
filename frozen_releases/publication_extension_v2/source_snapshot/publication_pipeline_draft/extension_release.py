#!/usr/bin/env python3
"""Verify a frozen publication-extension release and its live source mirror."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


class ExtensionReleaseError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_extension_release(release: Path, repo_root: Path) -> dict[str, Any]:
    """Fail unless the release and every snapshotted live source are identical."""
    release, repo_root = release.resolve(), repo_root.resolve()
    contents = release / "CONTENTS.sha256"
    manifest_path = release / "publication_extension_release_manifest.json"
    inventory_path = release / "source_inventory.csv"
    if not all(path.is_file() for path in (contents, manifest_path, inventory_path)):
        raise ExtensionReleaseError("Publication-extension release is incomplete.")
    for line in contents.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as error:
            raise ExtensionReleaseError(
                f"Malformed release checksum line: {line}"
            ) from error
        target = release / relative.removeprefix("./")
        if not target.is_file() or sha256(target) != expected:
            raise ExtensionReleaseError(f"Release checksum mismatch: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("release_status") != \
            "frozen_pre_external_test_publication_extension":
        raise ExtensionReleaseError("Release status is not pre-external-test frozen.")
    if manifest.get("holdout_accessed_by_freezer") is not False:
        raise ExtensionReleaseError("The extension freezer accessed a holdout.")
    with inventory_path.open(newline="", encoding="utf-8") as stream:
        inventory = list(csv.DictReader(stream))
    if len(inventory) != int(manifest.get("source_count", -1)):
        raise ExtensionReleaseError("Release source count is inconsistent.")
    for row in inventory:
        frozen = release / "source_snapshot" / row["path"]
        live = repo_root / row["path"]
        expected = row["sha256"]
        if not frozen.is_file() or sha256(frozen) != expected:
            raise ExtensionReleaseError(f"Frozen source mismatch: {row['path']}")
        if not live.is_file() or sha256(live) != expected:
            raise ExtensionReleaseError(
                f"Live source differs from frozen extension: {row['path']}"
            )
    result = dict(manifest)
    result["release_contents_sha256"] = sha256(contents)
    result["release_path"] = str(release)
    return result
