#!/usr/bin/env python3
"""Freeze prospective focused walk-forward code before panel/test execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from publication_pipeline_draft.focused_window_training_protocol import (
    validate_protocol,
)
from publication_pipeline_draft.freeze_training_release import deterministic_tar
from publication_pipeline_draft.publication_research_program import validate_program


class FocusedFreezeError(RuntimeError):
    pass


SOURCES = (
    "run_with_config.r", "evaluate_with_config.r", "config/config.yaml",
    "hpc/capture_publication_environment.sh",
    "hpc/validate_focused_walk_forward_v1.sh",
    "helper/load_data.r", "helper/time_split.r", "helper/marginals.r",
    "helper/synthetic_fidelity.r",
    "helper/reproducibility.r", "benchmark_models/dynamic_vine_NN.r",
    "rl/rl_environment.r", "rl/train_rl.r", "rl/training_sanity_check.r",
    "rl/evaluate_rl.r", "rl/synthetic_returns.r",
    "rl/revalidate_synthetic_bundle.r", "rl/action_projection.py",
    "rl/recurrent_baselines.py", "rl/policy_inference_server_v2.py",
    "publication_pipeline_draft/publication_research_program.py",
    "publication_pipeline_draft/asset_panel_protocol.py",
    "publication_pipeline_draft/walk_forward_windows.py",
    "publication_pipeline_draft/export_window_periods.py",
    "publication_pipeline_draft/materialize_window_return_input.py",
    "publication_pipeline_draft/extension_release.py",
    "publication_pipeline_draft/focused_seven_asset_panel.py",
    "publication_pipeline_draft/focused_walk_forward_windows.py",
    "publication_pipeline_draft/focused_window_training_protocol.py",
    "publication_pipeline_draft/freeze_focused_walk_forward_release.py",
    "publication_pipeline_draft/prepare_window_training_data.py",
    "publication_pipeline_draft/run_focused_window_sweep.py",
    "publication_pipeline_draft/audit_focused_window_sweep.py",
    "publication_pipeline_draft/generate_focused_window_policy_weights.py",
    "publication_pipeline_draft/generate_benchmark_weights.R",
    "publication_pipeline_draft/benchmark_weights.R",
    "publication_pipeline_draft/window_evaluation_protocol.py",
    "publication_pipeline_draft/score_focused_window.py",
    "publication_pipeline_draft/combine_focused_window_panels.py",
    "publication_pipeline_draft/analyze_focused_walk_forward.py",
    "publication_pipeline_draft/freeze_focused_walk_forward_results.py",
    "publication_pipeline_draft/publication_pipeline.py",
    "publication_pipeline_draft/daily_mark_to_market.py",
    "publication_pipeline_draft/build_window_realized_panel.R",
    "publication_pipeline_draft/config/publication_research_program_v2.json",
    "publication_pipeline_draft/config/focused_walk_forward_mechanisms_v1.json",
    "publication_pipeline_draft/config/benchmark_contract_v2.json",
    "publication_pipeline_draft/config/benchmark_contract_v3.json",
    "publication_pipeline_draft/config/benchmark_contract_v4.json",
    "publication_pipeline_draft/config/external_panel_metadata.example.json",
    "publication_pipeline_draft/FOCUSED_WALK_FORWARD_RUNBOOK.md",
    "publication_pipeline_draft/CAUSAL_RESULTS_DECISION_V1.md",
    "publication_pipeline_draft/tests/test_focused_walk_forward_protocol.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FocusedFreezeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(repo_root: Path, runtime: Path, output: Path,
           archive: Path | None) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    require(not output.exists() and
            (archive is None or not archive.exists()),
            "Focused release or archive already exists.")
    program_path = repo_root / (
        "publication_pipeline_draft/config/publication_research_program_v2.json")
    protocol_path = repo_root / (
        "publication_pipeline_draft/config/focused_walk_forward_mechanisms_v1.json")
    program = validate_program(program_path)
    protocol, protocol_sha256 = validate_protocol(protocol_path)
    require(runtime.is_dir() and (runtime / "CONTENTS.sha256").is_file(),
            "Hash-attested runtime inventory is required.")
    for line in (runtime / "CONTENTS.sha256").read_text(
            encoding="ascii").splitlines():
        if line.strip():
            expected, relative = line.split("  ", 1)
            target = runtime / relative.removeprefix("./")
            require(target.is_file() and sha256(target) == expected,
                    f"Runtime inventory mismatch: {target}")
    for relative in SOURCES:
        require((repo_root / relative).is_file(),
                f"Focused release source is missing: {relative}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        snapshot = temporary / "source_snapshot"
        inventory = []
        for relative in SOURCES:
            source = repo_root / relative
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            inventory.append({"path": relative, "sha256": sha256(destination),
                              "size_bytes": destination.stat().st_size})
        with (temporary / "source_inventory.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
            writer.writeheader()
            writer.writerows(inventory)
        shutil.copytree(runtime, temporary / "runtime_inventory")
        shutil.copy2(protocol_path, temporary / "focused_mechanism_protocol.json")
        manifest = {
            "schema_version": 1,
            # Reuse the generic extension verifier while making the focused
            # protocol identity explicit and immutable below.
            "release_status": "frozen_pre_external_test_publication_extension",
            "release_role": "focused_walk_forward_mechanism_v1",
            "program_id": program.raw["program_id"],
            "program_sha256": program.sha256,
            "focused_protocol_id": protocol["protocol_id"],
            "focused_protocol_sha256": protocol_sha256,
            "focused_experiment_count": 3,
            "focused_seed_count": 5,
            "source_count": len(inventory),
            "runtime_inventory_sha256": sha256(runtime / "CONTENTS.sha256"),
            "holdout_accessed_by_freezer": False,
            "consumed_holdout_reused": True,
            "confirmatory_claim_permitted_by_freeze": False,
            "evidence_class": "retrospective_walk_forward",
            "claim_limit": protocol["claim_limit"],
            "next_action": "materialize the two fixed retrospective windows without tuning",
        }
        (temporary / "publication_extension_release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (temporary / "READ_ONLY_RELEASE.txt").write_text(
            "Do not edit. Prospective retrospective mechanism protocol.\n",
            encoding="utf-8")
        checksum = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "CONTENTS.sha256":
                checksum.append(
                    f"{sha256(path)}  {path.relative_to(temporary).as_posix()}")
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(checksum) + "\n", encoding="ascii")
        os.replace(temporary, output)
        if archive is not None:
            archive.parent.mkdir(parents=True, exist_ok=True)
            deterministic_tar(output, archive)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    try:
        result = freeze(args.repo_root, args.runtime.resolve(), args.output,
                        args.archive)
    except (OSError, ValueError, KeyError, FocusedFreezeError) as error:
        print(f"FOCUSED WALK-FORWARD FREEZE FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
