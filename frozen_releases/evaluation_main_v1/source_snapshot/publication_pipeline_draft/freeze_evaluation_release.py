#!/usr/bin/env python3
"""Freeze the pre-holdout evaluation implementation and strategy contract.

The freezer always validates the frozen 20-policy full-model release. A
secondary no-vine release may be supplied only when it already exists and is
valid; it is never fabricated or required for the confirmatory main result.
The freezer checks the evaluation/benchmark mandates, snapshots every
executable evaluation source file, and never opens the realized holdout panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

try:  # package import in tests; script import in production CLI
    from .freeze_training_release import deterministic_tar, sha256_file, write_checksums
    from .publication_pipeline import Contract, ProtocolError
except ImportError:  # pragma: no cover - exercised by direct CLI invocation
    from freeze_training_release import deterministic_tar, sha256_file, write_checksums
    from publication_pipeline import Contract, ProtocolError


EVALUATION_SOURCES = (
    "publication_pipeline_draft/publication_pipeline.py",
    "publication_pipeline_draft/freeze_training_release.py",
    "publication_pipeline_draft/freeze_evaluation_release.py",
    "publication_pipeline_draft/benchmark_weights.R",
    "publication_pipeline_draft/generate_benchmark_weights.R",
    "publication_pipeline_draft/build_realized_panel.R",
    "publication_pipeline_draft/locked_evaluation_batch.py",
    "publication_pipeline_draft/config/evaluation_contract.json",
    "publication_pipeline_draft/config/benchmark_contract.json",
    "evaluate_with_config.r",
    "rl/evaluate_rl.r",
    "rl/rl_environment.r",
    "rl/action_projection.py",
    "benchmark_models/dynamic_vine_NN.r",
    "helper/load_data.r",
    "helper/time_split.r",
    "config/config.yaml",
    "tests/test_publication_benchmarks.r",
    "publication_pipeline_draft/tests/test_evaluation_extensions.py",
    "publication_pipeline_draft/PRE_HOLDOUT_EVALUATION_RUNBOOK.md",
)

BENCHMARK_IDS = (
    "equal_weight",
    "shrinkage_mean_variance",
    "dcc_garch",
    "static_vine",
    "rolling_vine",
    "dynamic_nn_vine",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"Required JSON file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected a JSON object: {path}")
    return value


def verify_contents_file(root: Path) -> None:
    checksum_file = root / "CONTENTS.sha256"
    if not checksum_file.is_file():
        raise ProtocolError(f"Frozen release lacks CONTENTS.sha256: {root}")
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ProtocolError(f"Frozen-release checksum mismatch: {path}")


def validate_training_release(
    root: Path, expected_seeds: int, expected_vine_mode: str | None
) -> list[dict[str, Any]]:
    root = root.resolve()
    if not root.is_dir():
        raise ProtocolError(f"Training release not found: {root}")
    verify_contents_file(root)
    manifest = load_json(root / "training_release_manifest.json")
    if manifest.get("release_status") != "frozen_pre_oos" or bool(
        manifest.get("holdout_accessed_by_freezer", True)
    ):
        raise ProtocolError(f"Training release is not a valid pre-OOS freeze: {root}")
    acceptance = manifest.get("acceptance", {})
    if int(acceptance.get("expected_seeds", -1)) != expected_seeds or acceptance.get(
        "decision"
    ) != "accepted_for_locked_evaluation":
        raise ProtocolError(f"Training release acceptance/count mismatch: {root}")

    seed_rows: list[dict[str, Any]] = []
    seed_dirs = sorted((root / "seeds").glob("seed_*"))
    if len(seed_dirs) != expected_seeds:
        raise ProtocolError(
            f"Expected {expected_seeds} seed directories in {root}; found {len(seed_dirs)}"
        )
    for directory in seed_dirs:
        try:
            seed = int(directory.name.removeprefix("seed_"))
        except ValueError as error:
            raise ProtocolError(f"Invalid seed directory: {directory}") from error
        checkpoint = directory / "td3_lstm_vine_full.pt"
        sanity = load_json(directory / "sanity_no_holdout/sanity_report.json")
        if not checkpoint.is_file() or not sanity.get("overall_pass") or not sanity.get(
            "publication_behavior_pass"
        ):
            raise ProtocolError(f"Seed {seed} lacks a passing full checkpoint/gate.")
        effective_vine_mode = expected_vine_mode or "full_legacy"
        if expected_vine_mode is not None:
            mode_file = directory / "vine_observation_mode.txt"
            reported_mode = sanity.get("vine_observation_mode")
            if expected_vine_mode == "full" and not mode_file.is_file():
                # Full-model sweeps frozen before the ablation protocol had no
                # mode marker because no zero-vine execution mode existed.
                # Preserve compatibility only for that one unambiguous legacy
                # case; the no-vine negative control remains strictly marked.
                if reported_mode not in (None, "full"):
                    raise ProtocolError(
                        f"Seed {seed} sanity report conflicts with legacy full mode."
                    )
                effective_vine_mode = "full_legacy"
            else:
                if (not mode_file.is_file()
                        or mode_file.read_text(encoding="utf-8").strip()
                        != expected_vine_mode):
                    raise ProtocolError(
                        f"Seed {seed} does not declare vine_observation_mode={expected_vine_mode}."
                    )
                if reported_mode != expected_vine_mode:
                    raise ProtocolError(
                        f"Seed {seed} sanity report was not run in {expected_vine_mode} mode."
                    )
                effective_vine_mode = expected_vine_mode
            if expected_vine_mode == "zero" and sanity.get(
                "no_vine_signal_mask"
            ) != "explicit_vine_and_scenario_cvar_v1":
                raise ProtocolError(
                    f"Seed {seed} exposes an invalid no-vine signal mask."
                )
        seed_rows.append(
            {
                "seed": seed,
                "release_directory": str(directory),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "vine_observation_mode": effective_vine_mode,
            }
        )
    return seed_rows


def compare_contracts(evaluation: Contract, benchmark: dict[str, Any]) -> None:
    mapping = {
        "evaluation_id": "evaluation_id",
        "net_exposure": "net_exposure",
        "gross_leverage": "gross_leverage",
        "max_long_weight": "max_long_weight",
        "max_short_weight": "max_short_weight",
        "weight_tolerance": "weight_tolerance",
        "turnover_cost": "turnover_cost",
        "annual_short_borrow_rate": "annual_short_borrow_rate",
        "annual_cash_borrow_rate": "annual_cash_borrow_rate",
        "crra_gamma": "crra_gamma",
    }
    for eval_name, benchmark_name in mapping.items():
        left, right = evaluation[eval_name], benchmark.get(benchmark_name)
        if isinstance(left, (int, float)):
            equal = right is not None and abs(float(left) - float(right)) <= 1e-12
        else:
            equal = left == right
        if not equal:
            raise ProtocolError(
                f"Evaluation and benchmark contracts disagree on {eval_name}: {left} != {right}"
            )
    if int(benchmark.get("schema_version", -1)) != 1:
        raise ProtocolError("Unsupported benchmark contract schema.")


def strategy_declaration(
    full_rows: list[dict[str, Any]], no_vine_rows: list[dict[str, Any]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in BENCHMARK_IDS:
        rows.append(
            {
                "strategy_id": method,
                "role": "benchmark",
                "seed": "",
                "ensemble_group": "",
                "vine_observation_mode": "not_applicable",
                "checkpoint_sha256": "not_applicable",
            }
        )
    for item in full_rows:
        rows.append(
            {
                "strategy_id": f"vine_td3_seed_{item['seed']}",
                "role": "proposed",
                "seed": item["seed"],
                "ensemble_group": "vine_td3_full_gate_pass",
                "vine_observation_mode": "full",
                "checkpoint_sha256": item["checkpoint_sha256"],
            }
        )
    for item in no_vine_rows:
        rows.append(
            {
                "strategy_id": f"no_vine_td3_seed_{item['seed']}",
                "role": "ablation",
                "seed": item["seed"],
                "ensemble_group": "no_vine_td3_gate_pass",
                "vine_observation_mode": "zero",
                "checkpoint_sha256": item["checkpoint_sha256"],
            }
        )
    return pd.DataFrame(rows)


def freeze_evaluation_release(
    repo_root: Path,
    full_training_release: Path,
    no_vine_training_release: Path | None,
    evaluation_contract: Path,
    benchmark_contract: Path,
    output: Path,
    bundle: Path | None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output = output.resolve()
    if output.exists():
        raise ProtocolError(f"Output already exists and will not be overwritten: {output}")
    full_rows = validate_training_release(full_training_release, 20, "full")
    no_vine_rows = (
        validate_training_release(no_vine_training_release, 10, "zero")
        if no_vine_training_release is not None else []
    )
    evaluation = Contract.read(evaluation_contract.resolve())
    benchmark = load_json(benchmark_contract.resolve())
    compare_contracts(evaluation, benchmark)

    missing = [relative for relative in EVALUATION_SOURCES if not (repo_root / relative).is_file()]
    if missing:
        raise ProtocolError(f"Evaluation source files are missing: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=output.parent))
    try:
        snapshot = temporary / "source_snapshot"
        inventory: list[dict[str, Any]] = []
        for relative in EVALUATION_SOURCES:
            source = repo_root / relative
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            inventory.append(
                {
                    "path": relative,
                    "size_bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
        inventory_frame = pd.DataFrame(inventory)
        inventory_frame.to_csv(temporary / "evaluation_source_inventory.csv", index=False)
        declaration = strategy_declaration(full_rows, no_vine_rows)
        declaration.to_csv(temporary / "strategy_declaration.csv", index=False)

        hash_payload = "\n".join(
            f"{row['sha256']}  {row['path']}" for row in inventory
        ).encode("utf-8")
        manifest = {
            "schema_version": 1,
            "release_status": "frozen_pre_holdout_evaluation",
            "holdout_accessed_by_freezer": False,
            "evaluation_id": evaluation["evaluation_id"],
            "full_policy_count": len(full_rows),
            "no_vine_policy_count": len(no_vine_rows),
            "benchmark_ids": list(BENCHMARK_IDS),
            "evaluation_source_count": len(inventory),
            "evaluation_code_contract_sha256": hashlib.sha256(hash_payload).hexdigest(),
            "full_training_release": str(full_training_release.resolve()),
            "no_vine_training_release": (
                str(no_vine_training_release.resolve())
                if no_vine_training_release is not None else None
            ),
            "secondary_ablations_included": bool(no_vine_rows),
            "next_action": "execute exactly one locked batch; do not edit this release",
        }
        (temporary / "evaluation_release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        (temporary / "READ_ONLY_RELEASE.txt").write_text(
            "Frozen before holdout access. Any change requires a new version and new hashes.\n",
            encoding="utf-8",
        )
        write_checksums(temporary)
        os.replace(temporary, output)
        if bundle is not None:
            deterministic_tar(output, bundle.resolve())
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--full-training-release", type=Path, required=True)
    parser.add_argument(
        "--no-vine-training-release", type=Path,
        help="Optional frozen 10-seed no-vine release; omit for main-only evaluation."
    )
    parser.add_argument(
        "--evaluation-contract", type=Path,
        default=Path("publication_pipeline_draft/config/evaluation_contract.json")
    )
    parser.add_argument(
        "--benchmark-contract", type=Path,
        default=Path("publication_pipeline_draft/config/benchmark_contract.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = freeze_evaluation_release(
        args.repo_root, args.full_training_release, args.no_vine_training_release,
        args.evaluation_contract, args.benchmark_contract, args.output, args.bundle
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
