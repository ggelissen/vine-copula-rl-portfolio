#!/usr/bin/env python3
"""Freeze publication-extension code and contracts before external testing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

try:
    from .causal_ablation_protocol import validated_rows
    from .publication_research_program import validate_program
except ImportError:  # direct script execution
    from causal_ablation_protocol import validated_rows
    from publication_research_program import validate_program


class FreezeError(RuntimeError):
    pass


SOURCES = (
    "run_with_config.r", "evaluate_with_config.r", "config/config.yaml",
    "hpc/capture_publication_environment.sh",
    "hpc/validate_publication_extension_v2.sh",
    "helper/reproducibility.r", "helper/load_data.r", "helper/time_split.r",
    "helper/marginals.r", "helper/timer.r",
    "benchmark_models/dynamic_vine_NN.r",
    "benchmark_models/expected_utility_single.r",
    "rl/rl_environment.r", "rl/train_rl.r", "rl/training_sanity_check.r",
    "rl/evaluate_rl.r", "rl/synthetic_returns.r",
    "rl/action_projection.py", "rl/recurrent_baselines.py",
    "rl/policy_inference_server.py", "rl/policy_inference_server_v2.py",
    "rl/generate_ablation_training_bundles.r",
    "publication_pipeline_draft/publication_research_program.py",
    "publication_pipeline_draft/asset_panel_protocol.py",
    "publication_pipeline_draft/walk_forward_windows.py",
    "publication_pipeline_draft/export_window_periods.py",
    "publication_pipeline_draft/materialize_window_return_input.py",
    "publication_pipeline_draft/extension_release.py",
    "publication_pipeline_draft/window_training_protocol.py",
    "publication_pipeline_draft/prepare_window_training_data.py",
    "publication_pipeline_draft/run_window_rl_sweep.py",
    "publication_pipeline_draft/audit_window_rl_sweep.py",
    "publication_pipeline_draft/generate_window_policy_weights.py",
    "publication_pipeline_draft/assemble_window_policy_ensembles.py",
    "publication_pipeline_draft/window_evaluation_protocol.py",
    "publication_pipeline_draft/aggregate_walk_forward_results.py",
    "publication_pipeline_draft/daily_mark_to_market.py",
    "publication_pipeline_draft/execute_window_evaluation.py",
    "publication_pipeline_draft/build_window_realized_panel.R",
    "eval/research_protocol.r", "eval/statistical_tests.r", "eval/ablation.r",
    "publication_pipeline_draft/benchmark_weights.R",
    "publication_pipeline_draft/extended_benchmark_weights.R",
    "publication_pipeline_draft/generate_benchmark_weights.R",
    "publication_pipeline_draft/generate_extended_benchmark_weights.R",
    "publication_pipeline_draft/causal_ablation_protocol.py",
    "publication_pipeline_draft/run_causal_sweep.py",
    "publication_pipeline_draft/merge_causal_sweep_status.py",
    "publication_pipeline_draft/audit_causal_sweep.py",
    "publication_pipeline_draft/freeze_training_release.py",
    "publication_pipeline_draft/freeze_publication_extension.py",
    "publication_pipeline_draft/future_confirmatory_protocol.py",
    "publication_pipeline_draft/publication_pipeline.py",
    "publication_pipeline_draft/config/publication_research_program_v2.json",
    "publication_pipeline_draft/config/benchmark_contract_v2.json",
    "publication_pipeline_draft/config/causal_ablation_contract_v2.json",
    "publication_pipeline_draft/config/external_panel_metadata.example.json",
    "publication_pipeline_draft/config/scalability_panel_metadata.example.json",
    "publication_pipeline_draft/config/scalability_universe_v1.json",
    "publication_pipeline_draft/PUBLICATION_EXTENSION_RUNBOOK.md",
    "tests/test_extended_publication_benchmarks.r",
    "tests/run_tests.r", "tests/test_publication_benchmarks.r",
    "tests/test_policy_process_isolation.r",
    "publication_pipeline_draft/tests/test_asset_panel_protocol.py",
    "publication_pipeline_draft/tests/test_publication_research_program.py",
    "publication_pipeline_draft/tests/test_causal_ablation_protocol.py",
    "publication_pipeline_draft/tests/test_causal_training_wiring.py",
    "publication_pipeline_draft/tests/test_walk_forward_windows.py",
    "publication_pipeline_draft/tests/test_materialize_window_return_input.py",
    "publication_pipeline_draft/tests/test_window_training_protocol.py",
    "publication_pipeline_draft/tests/test_window_evaluation_framework.py",
    "publication_pipeline_draft/tests/test_merge_causal_sweep_status.py",
    "publication_pipeline_draft/tests/test_publication_pipeline.py",
    "publication_pipeline_draft/tests/test_evaluator_accounting_v2.py",
    "publication_pipeline_draft/tests/test_external_training_wiring.py",
    "publication_pipeline_draft/tests/test_recurrent_baselines.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_contents(root: Path) -> None:
    checksum = root / "CONTENTS.sha256"
    for line in checksum.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as error:
            raise FreezeError(f"Malformed runtime checksum line: {line}") from error
        target = root / relative.removeprefix("./")
        if not target.is_file() or sha256(target) != expected:
            raise FreezeError(f"Runtime checksum mismatch: {target}")


def freeze(repo_root: Path, jobs: Path, runtime: Path, bundle_manifest: Path,
           output: Path, archive: Path | None) -> dict[str, object]:
    repo_root = repo_root.resolve()
    if output.exists() or (archive is not None and archive.exists()):
        raise FreezeError("Extension release/archive already exists.")
    program_path = repo_root / "publication_pipeline_draft/config/publication_research_program_v2.json"
    contract_path = repo_root / "publication_pipeline_draft/config/causal_ablation_contract_v2.json"
    program = validate_program(program_path)
    expected, contract_digest = validated_rows(
        contract_path, Path("data/publication_extension_runs_v2"))
    with jobs.open(newline="", encoding="utf-8") as stream:
        actual_jobs = list(csv.DictReader(stream))
    if len(actual_jobs) != len(expected) or len(actual_jobs) != 130:
        raise FreezeError("Causal job matrix is not the complete 130-job contract.")
    expected_keys = {(row["experiment_id"], str(row["seed"])) for row in expected}
    actual_keys = {(row["experiment_id"], row["seed"]) for row in actual_jobs}
    if expected_keys != actual_keys or any(
            row.get("contract_sha256") != contract_digest for row in actual_jobs):
        raise FreezeError("Causal job matrix does not match the current contract.")
    if not runtime.is_dir() or not (runtime / "CONTENTS.sha256").is_file():
        raise FreezeError("Runtime inventory must be a hash-attested directory.")
    verify_contents(runtime)
    if not bundle_manifest.is_file():
        raise FreezeError("Ablation bundle manifest was not found.")
    with bundle_manifest.open(newline="", encoding="utf-8") as stream:
        bundle_rows = list(csv.DictReader(stream))
    if {row.get("mode") for row in bundle_rows} != {
            "historical_prefix_repeated", "moving_block_bootstrap"}:
        raise FreezeError("Ablation bundle manifest does not contain both controls.")
    for row in bundle_rows:
        bundle_path = repo_root / row["file"]
        if not bundle_path.is_file() or sha256(bundle_path) != row["sha256"]:
            raise FreezeError(f"Ablation bundle hash mismatch: {bundle_path}")
    for relative in SOURCES:
        if not (repo_root / relative).is_file():
            raise FreezeError(f"Required extension source is missing: {relative}")

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
        shutil.copy2(jobs, temporary / "causal_jobs_v2.csv")
        shutil.copy2(bundle_manifest, temporary / "ablation_bundle_manifest.csv")
        shutil.copytree(runtime, temporary / "runtime_inventory")
        with (temporary / "source_inventory.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
            writer.writeheader(); writer.writerows(inventory)
        manifest = {
            "schema_version": 1,
            "release_status": "frozen_pre_external_test_publication_extension",
            "program_id": program.raw["program_id"],
            "program_sha256": program.sha256,
            "causal_contract_sha256": contract_digest,
            "causal_jobs_sha256": sha256(jobs),
            "causal_job_count": len(actual_jobs),
            "source_count": len(inventory),
            "runtime_inventory_sha256": sha256(runtime / "CONTENTS.sha256"),
            "ablation_bundle_manifest_sha256": sha256(bundle_manifest),
            "holdout_accessed_by_freezer": False,
            "consumed_holdout_reused": False,
            "confirmatory_claim_permitted_by_freeze": False,
            "next_action": "run development checks; use separate access ledger for future confirmation",
        }
        (temporary / "publication_extension_release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "READ_ONLY_RELEASE.txt").write_text(
            "Do not edit. This release freezes method/code, not a new confirmatory result.\n",
            encoding="utf-8")
        checksum_lines = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "CONTENTS.sha256":
                checksum_lines.append(f"{sha256(path)}  {path.relative_to(temporary).as_posix()}")
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="ascii")
        os.replace(temporary, output)
        if archive is not None:
            try:
                from .freeze_training_release import deterministic_tar
            except ImportError:
                from freeze_training_release import deterministic_tar
            archive.parent.mkdir(parents=True, exist_ok=True)
            deterministic_tar(output, archive)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--bundle-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    try:
        result = freeze(args.repo_root, args.jobs, args.runtime,
                        args.bundle_manifest, args.output, args.archive)
    except (RuntimeError, OSError, ValueError) as error:
        print(f"EXTENSION FREEZE FAILURE: {error}"); return 1
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
