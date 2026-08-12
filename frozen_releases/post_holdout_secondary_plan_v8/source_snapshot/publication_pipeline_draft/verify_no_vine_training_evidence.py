#!/usr/bin/env python3
"""Verify embedded no-vine metadata in completed training checkpoints.

This utility is intentionally read-only. It classifies both embedded metadata
and the older schema-5 case in which both mode fields are absent. Legacy
absence is not proof of zero-mode training; it requires independent manifest
and hash-verified source attestation before any loader may accept it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_MODE = "zero"
EXPECTED_MASK = "explicit_vine_and_scenario_cvar_v1"
CHECKPOINTS = ("td3_lstm_vine_pretrained.pt", "td3_lstm_vine_full.pt")


class EvidenceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, seeds: list[int]) -> dict[str, object]:
    import torch

    rows: list[dict[str, object]] = []
    for seed in seeds:
        directory = root / f"seed_{seed}"
        if not directory.is_dir():
            raise EvidenceError(f"Missing seed directory: {directory}")
        for checkpoint_name in CHECKPOINTS:
            checkpoint_path = directory / checkpoint_name
            if not checkpoint_path.is_file():
                raise EvidenceError(f"Missing checkpoint: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            architecture = checkpoint.get("architecture", {})
            mode = architecture.get("vine_observation_mode")
            signal_mask = architecture.get("no_vine_signal_mask")
            schema = architecture.get("checkpoint_schema")
            if (mode is None) != (signal_mask is None):
                raise EvidenceError(
                    f"{checkpoint_path}: partial mode metadata is not recoverable "
                    f"(mode={mode!r}, mask={signal_mask!r})"
                )
            legacy_missing = mode is None and signal_mask is None
            if mode not in (None, EXPECTED_MODE):
                raise EvidenceError(
                    f"{checkpoint_path}: vine_observation_mode={mode!r}, "
                    f"expected {EXPECTED_MODE!r}"
                )
            if signal_mask not in (None, EXPECTED_MASK):
                raise EvidenceError(
                    f"{checkpoint_path}: no_vine_signal_mask={signal_mask!r}, "
                    f"expected {EXPECTED_MASK!r}"
                )
            if int(schema) != 5:
                raise EvidenceError(
                    f"{checkpoint_path}: checkpoint_schema={schema!r}, expected 5"
                )
            rows.append(
                {
                    "seed": seed,
                    "checkpoint": checkpoint_name,
                    "checkpoint_schema": int(schema),
                    "vine_observation_mode": mode,
                    "no_vine_signal_mask": signal_mask,
                    "mode_metadata_status": (
                        "legacy_missing_requires_manifest_source_attestation"
                        if legacy_missing
                        else "embedded"
                    ),
                    "sha256": sha256_file(checkpoint_path),
                }
            )
    metadata_statuses = {str(row["mode_metadata_status"]) for row in rows}
    if len(metadata_statuses) != 1:
        raise EvidenceError(
            "Checkpoint set mixes embedded and legacy-missing mode metadata."
        )
    overall_status = (
        "valid_checkpoint_files_with_legacy_missing_mode_metadata"
        if metadata_statuses == {"legacy_missing_requires_manifest_source_attestation"}
        else "valid_embedded_no_vine_checkpoint_evidence"
    )
    return {
        "schema_version": 1,
        "status": overall_status,
        "evidence_is_post_hoc_recovery": True,
        "expected_mode": EXPECTED_MODE,
        "expected_signal_mask": EXPECTED_MASK,
        "seed_count": len(seeds),
        "checkpoint_count": len(rows),
        "torch_version": torch.__version__,
        "checkpoints": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated integers")
    parser.add_argument(
        "--require-embedded",
        action="store_true",
        help="Reject legacy checkpoints lacking embedded mode and mask metadata.",
    )
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise SystemExit("Seeds must be a non-empty unique comma-separated list.")
    try:
        payload = verify(args.sweep_root.resolve(), seeds)
    except EvidenceError as error:
        raise SystemExit(f"CHECKPOINT EVIDENCE FAILURE: {error}") from error
    if args.require_embedded and payload["status"] != (
        "valid_embedded_no_vine_checkpoint_evidence"
    ):
        raise SystemExit(
            "CHECKPOINT EVIDENCE FAILURE: clean retraining did not emit embedded "
            "no-vine checkpoint metadata"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
