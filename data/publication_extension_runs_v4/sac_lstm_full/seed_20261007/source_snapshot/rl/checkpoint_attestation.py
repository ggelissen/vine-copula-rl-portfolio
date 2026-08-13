"""Strict recovery of legacy checkpoint mode metadata from a hash-bound audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_MASK = "explicit_vine_and_scenario_cvar_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_architecture_mode(
    checkpoint_path: Path,
    architecture: dict[str, Any],
    expected_mode: str,
) -> tuple[dict[str, Any], str]:
    """Return mode-complete architecture or fail closed.

    Full-mode legacy checkpoints retain the historical default.  A zero-mode
    legacy checkpoint is accepted only when a per-seed repair record names the
    exact checkpoint file and SHA-256 and declares no scientific-model change.
    """

    checkpoint_path = checkpoint_path.resolve()
    actual = dict(architecture)
    mode = actual.get("vine_observation_mode")
    signal_mask = actual.get("no_vine_signal_mask")
    if (mode is None) != (signal_mask is None):
        raise RuntimeError("Checkpoint contains partial vine-mode metadata.")
    if mode is not None:
        return actual, "embedded"
    if expected_mode == "full":
        actual["vine_observation_mode"] = "full"
        return actual, "legacy_full_default"
    if expected_mode != "zero":
        raise RuntimeError(f"Unsupported expected vine mode: {expected_mode!r}")

    repair_path = checkpoint_path.parent / "vine_observation_mode_repair.json"
    if not repair_path.is_file():
        raise RuntimeError(
            "Zero-mode checkpoint lacks embedded metadata and a recovery attestation."
        )
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    if repair.get("repair_type") != (
        "post_hoc_missing_plaintext_mode_marker_reconstruction"
    ):
        raise RuntimeError("Checkpoint recovery attestation has an invalid repair type.")
    if repair.get("scientific_model_or_checkpoint_changed") is not False:
        raise RuntimeError("Checkpoint recovery attestation permits a model change.")
    if repair.get("reconstructed_value") != "zero":
        raise RuntimeError("Checkpoint recovery attestation does not declare zero mode.")
    evidence = repair.get("checkpoint_evidence")
    if not isinstance(evidence, list):
        raise RuntimeError("Checkpoint recovery attestation lacks checkpoint evidence.")
    matches = [
        row
        for row in evidence
        if isinstance(row, dict) and row.get("checkpoint") == checkpoint_path.name
    ]
    if len(matches) != 1:
        raise RuntimeError("Checkpoint recovery attestation is ambiguous.")
    row = matches[0]
    if row.get("sha256") != sha256_file(checkpoint_path):
        raise RuntimeError("Checkpoint hash differs from the recovery attestation.")
    if row.get("mode_metadata_status") != (
        "legacy_missing_requires_manifest_source_attestation"
    ):
        raise RuntimeError("Checkpoint was not classified as recoverable legacy metadata.")
    actual["vine_observation_mode"] = "zero"
    actual["no_vine_signal_mask"] = EXPECTED_MASK
    return actual, "attested_legacy_zero_mode"
