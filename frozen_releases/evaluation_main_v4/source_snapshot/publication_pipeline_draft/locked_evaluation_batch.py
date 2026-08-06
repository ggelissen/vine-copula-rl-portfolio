#!/usr/bin/env python3
"""Execute the single locked OOS batch from a frozen evaluation release.

Success and failure are both archived immutably.  A failed batch is evidence,
not permission to modify code and retry against the same holdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:  # package import in tests; script import in production CLI
    from .freeze_training_release import deterministic_tar, sha256_file
    from .publication_pipeline import MANIFEST_REQUIRED, ProtocolError
except ImportError:  # pragma: no cover - exercised by direct CLI invocation
    from freeze_training_release import deterministic_tar, sha256_file
    from publication_pipeline import MANIFEST_REQUIRED, ProtocolError


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"JSON file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return value


def verify_frozen_sources(repo_root: Path, release: Path) -> dict[str, Any]:
    contents = release / "CONTENTS.sha256"
    if not contents.is_file():
        raise ProtocolError("Frozen evaluation release lacks CONTENTS.sha256.")
    for line in contents.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        artifact = release / relative
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ProtocolError(f"Frozen evaluation checksum mismatch: {relative}")
    manifest = load_json(release / "evaluation_release_manifest.json")
    if manifest.get("release_status") != "frozen_pre_holdout_evaluation" or manifest.get(
        "holdout_accessed_by_freezer"
    ) is not False:
        raise ProtocolError("Evaluation release is not a valid pre-holdout freeze.")
    inventory_path = release / "evaluation_source_inventory.csv"
    if not inventory_path.is_file():
        raise ProtocolError("Frozen evaluation source inventory is missing.")
    with inventory_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(manifest.get("evaluation_source_count", -1)):
        raise ProtocolError("Evaluation source inventory count mismatch.")
    for row in rows:
        live = repo_root / row["path"]
        frozen = release / "source_snapshot" / row["path"]
        expected = row["sha256"]
        if not live.is_file() or not frozen.is_file():
            raise ProtocolError(f"Evaluation source is missing: {row['path']}")
        if sha256_file(live) != expected or sha256_file(frozen) != expected:
            raise ProtocolError(f"Live/frozen evaluation source mismatch: {row['path']}")
    return manifest


def seed_checkpoints(release: Path, expected: int, mode: str) -> list[dict[str, Any]]:
    directories = sorted((release / "seeds").glob("seed_*"))
    if len(directories) != expected:
        raise ProtocolError(f"Expected {expected} {mode} seeds; found {len(directories)}")
    rows = []
    for directory in directories:
        seed = int(directory.name.removeprefix("seed_"))
        checkpoint = directory / "td3_lstm_vine_full.pt"
        if not checkpoint.is_file():
            raise ProtocolError(f"Checkpoint missing: {checkpoint}")
        mode_file = directory / "vine_observation_mode.txt"
        if mode_file.is_file():
            if mode_file.read_text(encoding="utf-8").strip() != mode:
                raise ProtocolError(
                    f"Seed {seed} lacks vine_observation_mode={mode}."
                )
            recorded_mode = mode
        elif mode == "full":
            recorded_mode = "full_legacy"
        else:
            raise ProtocolError(
                f"Seed {seed} lacks vine_observation_mode={mode}."
            )
        rows.append(
            {
                "seed": seed,
                "directory": directory.resolve(),
                "checkpoint": checkpoint.resolve(),
                "checkpoint_sha256": sha256_file(checkpoint),
                "mode": recorded_mode,
            }
        )
    return rows


def run_logged(
    command: list[str], cwd: Path, env: dict[str, str], logs: Path, label: str
) -> float:
    start = time.monotonic()
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    elapsed = time.monotonic() - start
    (logs / f"{label}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (logs / f"{label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise ProtocolError(
            f"Locked command {label} failed with exit code {result.returncode}; "
            "see the immutable batch logs."
        )
    return elapsed


def manifest_row(**updates: Any) -> dict[str, Any]:
    row = {name: "" for name in MANIFEST_REQUIRED}
    row.update(
        completed="true", gate_pass="true", include_main="false",
        include_inference="false", report_seed_distribution="false",
        train_seconds="", evaluation_seconds="", notes=""
    )
    row.update(updates)
    return row


def create_strategy_manifest(
    path: Path,
    weights_dir: Path,
    benchmark_dir: Path,
    full_rows: list[dict[str, Any]],
    no_vine_rows: list[dict[str, Any]],
    timings: dict[str, float],
    frozen_hash: str,
) -> None:
    rows: list[dict[str, Any]] = [
        manifest_row(
            strategy_id="equal_weight", label="Equal weight", method="Equal weight",
            role="benchmark", include_main="true", include_inference="true",
            weight_log_path="GENERATE_EQUAL_WEIGHT",
            checkpoint_path="not_applicable", checkpoint_sha256="not_applicable",
            evaluation_seconds="0",
            notes="Generated by the common evaluator."
        )
    ]
    benchmark_labels = {
        "shrinkage_mean_variance": "Constrained shrinkage mean-variance",
        "dcc_garch": "DCC-GARCH",
        "static_vine": "Static vine optimizer",
        "rolling_vine": "Rolling vine optimizer",
        "dynamic_nn_vine": "Dynamic NN-vine optimizer without RL",
    }
    for strategy_id, label in benchmark_labels.items():
        weight = benchmark_dir / f"weights_{strategy_id}.csv"
        if not weight.is_file():
            raise ProtocolError(f"Benchmark weight log missing: {weight}")
        rows.append(
            manifest_row(
                strategy_id=strategy_id, label=label, method=label,
                role="benchmark", include_main="true", include_inference="true",
                weight_log_path=f"benchmark_weights/{weight.name}",
                weight_log_sha256=sha256_file(weight),
                checkpoint_path="not_applicable", checkpoint_sha256="not_applicable",
                evaluation_seconds=f"{timings.get('benchmarks', 0):.6f}",
                notes="Causal frozen benchmark; solver failures are fatal."
            )
        )
    for item in full_rows:
        run_name = item["directory"].name
        timing_key = f"full_{item['seed']}"
        weight = weights_dir / f"weights_rl_full_{run_name}.csv"
        if not weight.is_file():
            raise ProtocolError(f"Full-policy weight log missing: {weight}")
        rows.append(
            manifest_row(
                strategy_id=f"vine_td3_seed_{item['seed']}",
                label=f"NN-vine LSTM-TD3 seed {item['seed']}",
                method="NN-vine LSTM-TD3", seed=str(item["seed"]), role="proposed",
                ensemble_group="vine_td3_full_gate_pass",
                report_seed_distribution="true",
                weight_log_path=f"weights/{weight.name}", weight_log_sha256=sha256_file(weight),
                checkpoint_path=str(item["checkpoint"]),
                checkpoint_sha256=item["checkpoint_sha256"],
                config_sha256=frozen_hash, code_sha256=frozen_hash,
                evaluation_seconds=f"{timings[timing_key]:.6f}",
                notes="Passing frozen pre-holdout checkpoint."
            )
        )
    for item in no_vine_rows:
        run_name = item["directory"].name
        timing_key = f"no_vine_{item['seed']}"
        weight = weights_dir / f"weights_rl_full_{run_name}.csv"
        if not weight.is_file():
            raise ProtocolError(f"No-vine weight log missing: {weight}")
        rows.append(
            manifest_row(
                strategy_id=f"no_vine_td3_seed_{item['seed']}",
                label=f"No-vine LSTM-TD3 seed {item['seed']}",
                method="No-vine LSTM-TD3", seed=str(item["seed"]), role="ablation",
                ensemble_group="no_vine_td3_gate_pass",
                report_seed_distribution="true",
                weight_log_path=f"weights/{weight.name}", weight_log_sha256=sha256_file(weight),
                checkpoint_path=str(item["checkpoint"]),
                checkpoint_sha256=item["checkpoint_sha256"],
                config_sha256=frozen_hash, code_sha256=frozen_hash,
                evaluation_seconds=f"{timings[timing_key]:.6f}",
                notes="Matched-capacity zero-vine-state ablation."
            )
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_REQUIRED)
        writer.writeheader()
        writer.writerows(rows)


def execute_batch(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    release = args.evaluation_release.resolve()
    output = args.output.resolve()
    bundle = args.bundle.resolve()
    if output.exists() or bundle.exists() or bundle.with_suffix(bundle.suffix + ".sha256").exists():
        raise ProtocolError("Locked batch output/bundle already exists; refusing to overwrite.")
    frozen = verify_frozen_sources(repo_root, release)
    full_release = Path(frozen["full_training_release"]).resolve()
    no_vine_release_value = frozen.get("no_vine_training_release")
    full_rows = seed_checkpoints(full_release, 20, "full")
    no_vine_rows = (
        seed_checkpoints(Path(no_vine_release_value).resolve(), 10, "zero")
        if no_vine_release_value else []
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=output.parent))
    logs = temporary / "command_logs"
    weights = temporary / "weights"
    logs.mkdir(); weights.mkdir()
    holdout_access_started = False
    timings: dict[str, float] = {}
    base_env = os.environ.copy()
    base_env.update({"LC_ALL": "C", "LANG": "C", "LANGUAGE": "C", "TZ": "UTC"})
    try:
        holdout_access_started = True
        inputs = temporary / "inputs"
        timings["realized_panel"] = run_logged(
            [args.rscript, "--vanilla", "publication_pipeline_draft/build_realized_panel.R",
             "config/config.yaml", str(inputs)],
            repo_root, base_env, logs, "build_realized_panel")

        for group, rows, mode in (("full", full_rows, "full"),
                                  ("no_vine", no_vine_rows, "zero")):
            for item in rows:
                env = base_env.copy()
                env.update(
                    {
                        "EVAL_MODEL_DIR": str(item["directory"]),
                        "EVAL_OUTPUT_DIR": str(weights),
                        "EVAL_WEIGHTS_ONLY": "true",
                        "VINE_OBSERVATION_MODE": mode,
                        "EVAL_WINDOW_ID": str(frozen["evaluation_id"]),
                    }
                )
                label = f"{group}_{item['seed']}"
                timings[label] = run_logged(
                    [args.rscript, "--vanilla", "evaluate_with_config.r",
                     "config/config.yaml", str(item["directory"])],
                    repo_root, env, logs, label)

        benchmark_dir = temporary / "benchmark_weights"
        timings["benchmarks"] = run_logged(
            [args.rscript, "--vanilla",
             "publication_pipeline_draft/generate_benchmark_weights.R",
             "config/config.yaml",
             "publication_pipeline_draft/config/benchmark_contract.json",
             str(benchmark_dir)],
            repo_root, base_env, logs, "benchmarks")

        strategy_manifest = temporary / "strategy_manifest.csv"
        create_strategy_manifest(
            strategy_manifest, weights, benchmark_dir, full_rows, no_vine_rows,
            timings, str(frozen["evaluation_code_contract_sha256"])
        )
        results = temporary / "publication_results"
        timings["common_evaluator"] = run_logged(
            [sys.executable, "publication_pipeline_draft/publication_pipeline.py",
             "--contract", "publication_pipeline_draft/config/evaluation_contract.json",
             "--realized", str(inputs / "realized_asset_gross.csv"),
             "--strategies", str(strategy_manifest), "--output", str(results)],
            repo_root, base_env, logs, "common_evaluator")
        status = {
            "schema_version": 1,
            "status": "complete",
            "holdout_accessed": True,
            "evaluation_release_sha256": frozen["evaluation_code_contract_sha256"],
            "full_policy_count": len(full_rows),
            "no_vine_policy_count": len(no_vine_rows),
            "benchmark_count": 6,
            "policy_inference_protocol": "file_ipc_isolated_libtorch_v1",
            "policy_python": base_env.get("POLICY_PYTHON", ""),
            "timings_seconds": timings,
        }
        (temporary / "locked_batch_manifest.json").write_text(
            json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, output)
        deterministic_tar(output, bundle)
        return status
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "holdout_accessed": holdout_access_started,
            "error_type": type(error).__name__,
            "error": str(error),
            "timings_seconds": timings,
            "scientific_note": (
                "This failed locked batch is preserved. Do not tune or retry after holdout access "
                "without declaring a new confirmatory sample/protocol."
            ),
        }
        (temporary / "locked_batch_manifest.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, output)
        deterministic_tar(output, bundle)
        raise ProtocolError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evaluation-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--rscript", default="Rscript")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = execute_batch(args)
    except ProtocolError as error:
        print(f"LOCKED BATCH FAILURE: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
