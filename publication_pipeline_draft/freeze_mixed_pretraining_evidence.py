#!/usr/bin/env python3
"""Register the completed mixed-pretraining evidence as an immutable release.

The operation is byte preserving.  It verifies both supplied archive sidecars,
checks the internal analysis/audit checksum inventories, extracts only the
publication-relevant evidence, and records that the consumed holdout cannot be
used for another confirmatory claim.
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
from pathlib import Path, PurePosixPath
from typing import Any


class MixedEvidenceFreezeError(RuntimeError):
    """Raised when the supplied evidence cannot be frozen safely."""


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RESULT_PREFIX = "analysis_outputs/mixed_pretraining_response_v1_results/"
WEIGHT_PREFIX = "analysis_outputs/mixed_pretraining_response_v1_weights/"
AUDIT_PREFIX = "analysis_outputs/mixed_pretraining_response_v1_audit/"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MixedEvidenceFreezeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_digest(sidecar: Path, archive: Path) -> str:
    require(sidecar.is_file(), f"Checksum sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    require(bool(fields) and SHA256_RE.fullmatch(fields[0].lower()) is not None,
            f"Malformed SHA-256 sidecar: {sidecar}")
    if len(fields) > 1:
        require(Path(fields[-1].lstrip("* ")).name == archive.name,
                f"Checksum sidecar names another archive: {sidecar}")
    return fields[0].lower()


def safe_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    result: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        require(not path.is_absolute() and ".." not in path.parts,
                f"Unsafe archive member: {member.name}")
        require(not (member.issym() or member.islnk() or member.isdev()),
                f"Links and devices are forbidden: {member.name}")
        name = path.as_posix().lstrip("./")
        require(name not in result, f"Duplicate archive member: {name}")
        result[name] = member
    return result


def member_bytes(archive: tarfile.TarFile,
                 members: dict[str, tarfile.TarInfo], name: str) -> bytes:
    member = members.get(name)
    require(member is not None and member.isfile(),
            f"Required archive member is missing: {name}")
    stream = archive.extractfile(member)
    require(stream is not None, f"Could not read archive member: {name}")
    return stream.read()


def verify_inventory(archive: tarfile.TarFile,
                     members: dict[str, tarfile.TarInfo], prefix: str) -> int:
    inventory_name = f"{prefix}CONTENTS.sha256"
    inventory = member_bytes(archive, members, inventory_name).decode("ascii")
    verified = 0
    for line_number, line in enumerate(inventory.splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
        require(match is not None,
                f"Malformed {inventory_name} line {line_number}")
        expected, relative = match.group(1).lower(), match.group(2)
        relative_path = PurePosixPath(relative)
        require(not relative_path.is_absolute() and ".." not in relative_path.parts,
                f"Unsafe checksum path: {relative}")
        actual = hashlib.sha256(member_bytes(
            archive, members, f"{prefix}{relative_path.as_posix()}"
        )).hexdigest()
        require(actual == expected, f"Internal checksum mismatch: {prefix}{relative}")
        verified += 1
    require(verified > 0, f"Empty internal checksum inventory: {inventory_name}")
    return verified


def extract_selected(archive: tarfile.TarFile,
                     members: dict[str, tarfile.TarInfo], output: Path) -> int:
    selected = 0
    for name, member in members.items():
        include = (
            name.startswith(RESULT_PREFIX)
            or name.startswith(AUDIT_PREFIX)
            or name == f"{WEIGHT_PREFIX}mixed_pretraining_comparison_weight_manifest.csv"
            or name.startswith(f"{WEIGHT_PREFIX}weights/")
        )
        if not include or not member.isfile():
            continue
        if name.startswith(RESULT_PREFIX):
            relative = Path("analysis_results") / name.removeprefix(RESULT_PREFIX)
        elif name.startswith(AUDIT_PREFIX):
            relative = Path("checkpoint_audit") / name.removeprefix(AUDIT_PREFIX)
        else:
            relative = Path("weight_evidence") / name.removeprefix(WEIGHT_PREFIX)
        target = output / "source_evidence" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(member_bytes(archive, members, name))
        selected += 1
    require(selected >= 20, "Too few publication-relevant members were extracted")
    return selected


def validate_final_archive(path: Path) -> tuple[dict[str, Any], int, int]:
    with tarfile.open(path, "r:gz") as archive:
        members = safe_members(archive)
        results_verified = verify_inventory(archive, members, RESULT_PREFIX)
        audit_verified = verify_inventory(archive, members, AUDIT_PREFIX)
        analysis = json.loads(member_bytes(
            archive, members,
            f"{RESULT_PREFIX}mixed_pretraining_analysis_manifest.json"
        ).decode("utf-8"))
        require(analysis.get("status") ==
                "mixed_pretraining_four_arm_analysis_complete",
                "Mixed-pretraining analysis is not complete")
        require(analysis.get("evidence_class") == "post_holdout_explanatory",
                "Unexpected evidence class")
        require(analysis.get("confirmatory_claim_permitted") is False,
                "Consumed-holdout evidence cannot permit confirmation")
        require(analysis.get("same_holdout_further_tuning_authorized") is False,
                "Further same-holdout tuning must be prohibited")
        require(int(analysis.get("arm_count", 0)) == 4 and
                int(analysis.get("common_complete_periods", 0)) == 22,
                "Four-arm primary sample declaration is invalid")
        audit = json.loads(member_bytes(
            archive, members,
            f"{AUDIT_PREFIX}mixed_pretraining_audit_manifest.json"
        ).decode("utf-8"))
        require(audit.get("status") ==
                "mixed_pretraining_comparison_audit_passed",
                "Checkpoint audit did not pass")
        require(all(audit.get(field) is True for field in (
            "all_behavior_gate_enforcement_valid",
            "all_checkpoint_metadata_match", "all_checkpoint_tensors_finite")),
            "Checkpoint audit contains a failed integrity condition")
        return analysis, results_verified, audit_verified


def checkpoint_inventory(path: Path) -> dict[str, int]:
    with tarfile.open(path, "r:gz") as archive:
        members = safe_members(archive)
    names = list(members)
    result = {
        "member_count": len(names),
        "pretrained_checkpoint_count": sum(
            name.endswith("_pretrained.pt") for name in names),
        "full_checkpoint_count": sum(name.endswith("_full.pt") for name in names),
        "behavior_gate_file_count": sum(
            name.endswith("pretraining_behavior_gate.csv") for name in names),
    }
    require(result["pretrained_checkpoint_count"] == 10,
            "Checkpoint archive must contain ten mixed pretrained checkpoints")
    require(result["full_checkpoint_count"] == 10,
            "Checkpoint archive must contain ten mixed full checkpoints")
    require(result["behavior_gate_file_count"] == 10,
            "Checkpoint archive must contain ten behavior-gate files")
    return result


def freeze(final_archive: Path, final_sidecar: Path,
           checkpoint_archive: Path, checkpoint_sidecar: Path,
           output: Path) -> dict[str, Any]:
    paths = (final_archive, checkpoint_archive)
    require(all(path.is_file() for path in paths), "A supplied archive is missing")
    require(not output.exists(), f"Frozen output already exists: {output}")
    final_expected = sidecar_digest(final_sidecar, final_archive)
    checkpoint_expected = sidecar_digest(checkpoint_sidecar, checkpoint_archive)
    require(sha256(final_archive) == final_expected,
            "Final archive SHA-256 mismatch")
    require(sha256(checkpoint_archive) == checkpoint_expected,
            "Checkpoint archive SHA-256 mismatch")
    analysis, results_verified, audit_verified = validate_final_archive(final_archive)
    checkpoints = checkpoint_inventory(checkpoint_archive)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.copyfile(final_archive, temporary / final_archive.name)
        shutil.copyfile(checkpoint_archive, temporary / checkpoint_archive.name)
        (temporary / f"{final_archive.name}.sha256").write_text(
            f"{final_expected}  {final_archive.name}\n", encoding="ascii")
        (temporary / f"{checkpoint_archive.name}.sha256").write_text(
            f"{checkpoint_expected}  {checkpoint_archive.name}\n", encoding="ascii")
        with tarfile.open(final_archive, "r:gz") as archive:
            selected = extract_selected(archive, safe_members(archive), temporary)
        manifest = {
            "schema_version": 1,
            "status": "frozen_mixed_pretraining_evidence_v1",
            "operation": "byte_exact_registration_and_selective_extraction",
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "same_holdout_further_tuning_authorized": False,
            "final_archive": final_archive.name,
            "final_archive_sha256": final_expected,
            "checkpoint_archive": checkpoint_archive.name,
            "checkpoint_archive_sha256": checkpoint_expected,
            "checkpoint_inventory": checkpoints,
            "results_inventory_verified_files": results_verified,
            "audit_inventory_verified_files": audit_verified,
            "selected_evidence_files": selected,
            "source_analysis_manifest": analysis,
            "scientific_note": (
                "This is terminal same-holdout, post-selection explanatory evidence. "
                "Freezing preserves it but cannot convert it into confirmation."),
        }
        (temporary / "mixed_pretraining_evidence_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "READ_ONLY_RELEASE.txt").write_text(
            "Immutable post-holdout explanatory evidence. Do not edit in place.\n",
            encoding="utf-8")
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "CONTENTS.sha256").write_text("".join(
            f"{sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
            for path in files), encoding="ascii")
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-archive", required=True, type=Path)
    parser.add_argument("--final-checksum", required=True, type=Path)
    parser.add_argument("--checkpoint-archive", required=True, type=Path)
    parser.add_argument("--checkpoint-checksum", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = freeze(
            args.final_archive.resolve(), args.final_checksum.resolve(),
            args.checkpoint_archive.resolve(), args.checkpoint_checksum.resolve(),
            args.output.resolve())
    except (MixedEvidenceFreezeError, OSError, tarfile.TarError,
            json.JSONDecodeError) as error:
        print(f"MIXED EVIDENCE FREEZE FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
