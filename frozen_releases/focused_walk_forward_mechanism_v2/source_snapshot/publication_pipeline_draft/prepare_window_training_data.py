#!/usr/bin/env python3
"""Execute and verify one frozen window's development-only data generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from publication_pipeline_draft.extension_release import (
    ExtensionReleaseError, verify_extension_release,
)


class PreparationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_contract(root: Path) -> tuple[dict, dict]:
    broad_manifest = root / "window_training_manifest.json"
    focused_manifest = root / "focused_window_training_manifest.json"
    manifest_candidates = [path for path in (broad_manifest, focused_manifest)
                           if path.is_file()]
    if len(manifest_candidates) != 1:
        raise PreparationError(
            "Window contract must contain exactly one broad or focused manifest.")
    manifest_path = manifest_candidates[0]
    environment_path = root / "generator_environment.json"
    jobs_path = (root / "window_rl_jobs.csv" if manifest_path == broad_manifest
                 else root / "focused_window_jobs.csv")
    contents = root / "CONTENTS.sha256"
    if not all(path.is_file() for path in
               (manifest_path, environment_path, jobs_path, contents)):
        raise PreparationError("Window training contract is incomplete.")
    for line in contents.read_text(encoding="ascii").splitlines():
        if line.strip():
            expected, relative = line.split("  ", 1)
            target = root / relative
            if not target.is_file() or sha256(target) != expected:
                raise PreparationError(f"Contract checksum mismatch: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("release_status") not in {
            "frozen_development_window_training_contract",
            "frozen_focused_window_training_contract"}:
        raise PreparationError("Contract status does not authorize generation.")
    if manifest.get("confirmatory_claim_permitted") is not False:
        raise PreparationError("Development generator cannot authorize confirmation.")
    if manifest.get("jobs_sha256") != sha256(jobs_path):
        raise PreparationError("Frozen job matrix hash mismatch.")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    return manifest, environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--rscript", required=True, type=Path)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--sim-cores", type=int, default=8)
    parser.add_argument(
        "--adopt-existing-revalidated", action="store_true",
        help=("Verify and attest an existing sampling-aware revalidated bundle "
              "without rerunning synthetic simulation."))
    args = parser.parse_args()
    try:
        repo = args.repo_root.resolve()
        release = verify_extension_release(args.release, repo)
        manifest, generator_environment = verify_contract(args.contract.resolve())
        if manifest.get("program_sha256") != release.get("program_sha256"):
            raise PreparationError("Window contract and extension release differ.")
        if manifest.get("release_status") == \
                "frozen_focused_window_training_contract":
            if (release.get("release_role") !=
                    "focused_walk_forward_mechanism_v1" or
                    release.get("focused_protocol_sha256") !=
                    manifest.get("focused_protocol_sha256")):
                raise PreparationError(
                    "Focused generator contract and prospective release differ.")
        if not args.rscript.is_file() or args.sim_cores < 1:
            raise PreparationError("Rscript/simulation core setting is invalid.")
        if args.log_root.exists():
            raise PreparationError("Immutable generator log path already exists.")
        bundle = repo / generator_environment["SYNTHETIC_RETURNS_FILE"]
        sidecar = repo / generator_environment["SYNTHETIC_BUNDLE_MANIFEST"]
        if args.adopt_existing_revalidated:
            if not bundle.is_file() or not sidecar.is_file():
                raise PreparationError(
                    "Revalidated bundle adoption requires the existing bundle and sidecar.")
        elif bundle.exists() or sidecar.exists():
            raise PreparationError("Window training data already exists.")
        args.log_root.mkdir(parents=True)
        if not args.adopt_existing_revalidated:
            environment = os.environ.copy()
            environment.update({key: str(value) for key, value in
                                generator_environment.items()})
            environment.update({
                "VINE_SIM_CORES": str(args.sim_cores),
                "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C", "TZ": "UTC",
            })
            command = [str(args.rscript), "--vanilla", "rl/synthetic_returns.r",
                       str(args.config)]
            stdout_path = args.log_root / "generator.stdout.txt"
            stderr_path = args.log_root / "generator.stderr.txt"
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                result = subprocess.run(command, cwd=repo, env=environment,
                                        stdout=stdout, stderr=stderr, check=False)
            if result.returncode:
                raise PreparationError(
                    f"Synthetic generator failed with exit {result.returncode}; inspect {args.log_root}.")
        if not bundle.is_file() or not sidecar.is_file():
            raise PreparationError("Generator completed without attested bundle outputs.")
        evidence = json.loads(sidecar.read_text(encoding="utf-8"))
        checks = {
            "bundle_sha256": sha256(bundle),
            "window_id": manifest["window_id"],
            "panel_id": manifest["panel_id"],
            "asset_count": manifest["asset_count"],
            "vine_truncation_level": manifest["vine_truncation_level"],
            "diagnostics_passed": True,
            "confirmatory_claim_permitted": False,
        }
        mismatches = {key: (evidence.get(key), expected)
                      for key, expected in checks.items()
                      if evidence.get(key) != expected}
        if mismatches:
            raise PreparationError(f"Generated bundle attestation mismatch: {mismatches}")
        if args.adopt_existing_revalidated:
            adoption_checks = {
                "diagnostic_gate_protocol": "sampling_aware_guardrailed_v2",
                "synthetic_returns_regenerated": False,
                "confirmatory_claim_permitted": False,
            }
            adoption_mismatches = {
                key: (evidence.get(key), expected)
                for key, expected in adoption_checks.items()
                if evidence.get(key) != expected
            }
            if adoption_mismatches or not evidence.get("parent_bundle_sha256"):
                raise PreparationError(
                    "Existing bundle is not an attested no-resimulation "
                    f"revalidation: {adoption_mismatches}")
        result_manifest = {
            "schema_version": 1,
            "status": "window_training_data_ready",
            "window_id": manifest["window_id"],
            "bundle_sha256": sha256(bundle),
            "finetune_episodes": int(evidence["finetune_episodes"]),
            "pretrain_episodes": int(evidence["pretrain_episodes"]),
            "test_data_used_for_training": False,
            "diagnostic_gate_protocol": evidence.get(
                "diagnostic_gate_protocol", "strict_v1"),
            "synthetic_returns_regenerated": evidence.get(
                "synthetic_returns_regenerated", True),
            "existing_bundle_adopted": args.adopt_existing_revalidated,
            "confirmatory_claim_permitted": False,
        }
        (args.log_root / "preparation_manifest.json").write_text(
            json.dumps(result_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result_manifest, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, PreparationError,
            ExtensionReleaseError) as error:
        print(f"WINDOW DATA PREPARATION FAILURE: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
