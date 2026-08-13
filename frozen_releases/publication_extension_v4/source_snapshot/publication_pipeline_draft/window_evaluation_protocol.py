#!/usr/bin/env python3
"""Assemble a fail-closed common evaluator contract for one external window."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from publication_pipeline_draft.extension_release import (
    ExtensionReleaseError, verify_extension_release,
)
from publication_pipeline_draft.run_window_rl_sweep import verify_contract


class WindowEvaluationError(RuntimeError):
    pass


BENCHMARK_LABELS = {
    "equal_weight": "Equal weight",
    "minimum_variance": "Minimum variance",
    "risk_parity": "Long-only risk parity",
    "shrinkage_mean_variance": "Shrinkage mean-variance",
    "mean_cvar": "Empirical mean-CVaR",
    "momentum_tilt": "12-1 momentum tilt",
    "black_litterman_momentum_views": "Black-Litterman momentum views",
    "dcc_garch": "DCC-GARCH",
    "static_vine": "Static vine optimizer",
    "rolling_vine": "Rolling vine optimizer",
    "dynamic_nn_vine": "Dynamic NN-vine optimizer",
}
ALGORITHM_LABELS = {
    "td3": "NN-vine LSTM-TD3",
    "ddpg": "NN-vine LSTM-DDPG",
    "sac": "NN-vine LSTM-SAC",
    "ppo": "NN-vine LSTM-PPO",
    "a2c": "NN-vine LSTM-A2C",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WindowEvaluationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file(), f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def verify_contents(root: Path) -> None:
    contents = root / "CONTENTS.sha256"
    require(contents.is_file(), f"CONTENTS.sha256 not found: {root}")
    for line in contents.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = root / relative.removeprefix("./")
        require(target.is_file() and sha256(target) == expected,
                f"Checksum mismatch: {target}")


def resolve_recorded(path: str, repo_root: Path, inventory: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    candidates = [(repo_root / value).resolve(),
                  (inventory.parent / value).resolve()]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    require(len(existing) == 1 or
            (len(existing) == 2 and existing[0] == existing[1]),
            f"Recorded artifact path is missing or ambiguous: {path}")
    return existing[0]


def relative(path: Path, future_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), future_root.resolve())).as_posix()


def verify_benchmark_family(root: Path, expected: set[str],
                            allowed_codes: set[int],
                            tolerance: float) -> dict[str, Path]:
    fields, manifest = read_csv(root / "benchmark_manifest.csv")
    require({row.get("method") for row in manifest} == expected,
            f"Benchmark family differs from the protocol: {root}")
    require(len(manifest) == len(expected), "Duplicate benchmark manifest rows.")
    files: dict[str, Path] = {}
    for row in manifest:
        method = row["method"]
        path = root / row["weight_file"]
        require(path.is_file(), f"Benchmark weights missing: {path}")
        _, weights = read_csv(path)
        require(len(weights) == 24, f"Benchmark {method} must have 24 rows.")
        files[method] = path.resolve()
    _, audit = read_csv(root / "solver_audit.csv")
    audit_methods = {row.get("method") for row in audit}
    require(expected <= audit_methods,
            f"Solver audit omits a benchmark in {root}")
    counts = {method: 0 for method in expected}
    missing_tokens = {"", "na", "nan", "null", "none"}
    for row in audit:
        method = row.get("method")
        if method not in expected:
            continue
        counts[method] += 1
        decision, latest = row.get("decision_date", ""), row.get("latest_input_date", "")
        require(bool(decision) and (not latest or latest <= decision),
                f"Future-data audit failure for {method} at {decision}")
        convergence = row.get("convergence", "").strip()
        if convergence.lower() not in missing_tokens:
            require(int(float(convergence)) in allowed_codes,
                    f"Unaccepted convergence code for {method}: {convergence}")
        residual = row.get("constraint_residual", "").strip()
        if residual.lower() not in missing_tokens:
            require(math.isfinite(float(residual)) and float(residual) <= tolerance,
                    f"Constraint residual failure for {method}: {residual}")
        text = " ".join(row.values()).lower()
        require("silent fallback" not in text and "fallback_used" not in text,
                f"Fallback marker found in solver audit for {method}")
    require(all(count == 24 for count in counts.values()),
            f"Every benchmark needs 24 solver-audit rows: {counts}")
    return files


def materialize(repo_root: Path, release_root: Path, window_contract_root: Path,
                checkpoint_audit_root: Path, policy_inventory_path: Path,
                core_benchmark_root: Path, extended_benchmark_root: Path,
                realized_panel_path: Path, benchmark_contract_path: Path,
                output: Path) -> dict[str, Any]:
    require(not output.exists(), f"Output already exists: {output}")
    repo_root = repo_root.resolve()
    release = verify_extension_release(release_root, repo_root)
    window_contract, jobs = verify_contract(window_contract_root)
    require(window_contract.get("program_sha256") == release.get("program_sha256"),
            "Window and extension contracts use different research programs.")
    verify_contents(checkpoint_audit_root)
    audit_manifest_path = checkpoint_audit_root / "window_sweep_audit_manifest.json"
    audit = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    require(audit.get("status") == "window_rl_sweep_audit_passed" and
            audit.get("job_count") == 50 and
            audit.get("window_id") == window_contract["window_id"],
            "Checkpoint audit is not the exact passed window sweep.")
    _, checkpoint_rows = read_csv(checkpoint_audit_root / "checkpoint_audit.csv")
    checkpoint_by_key = {(row["algorithm"], row["seed"]): row
                         for row in checkpoint_rows}
    require(len(checkpoint_by_key) == 50,
            "Checkpoint audit needs five algorithms by ten seeds.")

    _, policy_rows = read_csv(policy_inventory_path)
    policy_by_key = {(row["algorithm"], row["seed"]): row for row in policy_rows}
    require(set(policy_by_key) == set(checkpoint_by_key),
            "Policy logs do not exactly match audited checkpoints.")
    policy_paths: dict[tuple[str, str], Path] = {}
    for key, row in policy_by_key.items():
        path = resolve_recorded(row["weight_file"], repo_root,
                                policy_inventory_path)
        require(sha256(path) == row["sha256"],
                f"Policy weight hash mismatch: {path}")
        _, values = read_csv(path)
        require(len(values) == 24, f"Policy log does not have 24 rows: {path}")
        policy_paths[key] = path

    benchmark_contract = json.loads(
        benchmark_contract_path.read_text(encoding="utf-8"))
    allowed = {int(value) for value in
               benchmark_contract["optimizer_allowed_convergence_codes"]}
    tolerance = max(float(benchmark_contract["weight_tolerance"]), 1e-8)
    core_expected = {"equal_weight", "shrinkage_mean_variance", "dcc_garch",
                     "static_vine", "rolling_vine", "dynamic_nn_vine"}
    extended_expected = set(BENCHMARK_LABELS) - core_expected
    benchmark_paths = verify_benchmark_family(
        core_benchmark_root, core_expected, allowed, tolerance)
    benchmark_paths.update(verify_benchmark_family(
        extended_benchmark_root, extended_expected, allowed, tolerance))

    realized_fields, realized_rows = read_csv(realized_panel_path)
    weight_assets = [name[2:] for name in realized_fields if name.startswith("g_")]
    require(len(realized_rows) == 24 and len(weight_assets) ==
            int(window_contract["asset_count"]),
            "Realized panel dimensions differ from the window contract.")
    require({row["window_id"] for row in realized_rows} ==
            {window_contract["window_id"]},
            "Realized panel uses the wrong window identifier.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        final_root = output.resolve()
        contract = {
            "schema_version": 1,
            "evaluation_id": f"external_development_{window_contract['window_id']}",
            "evidence_class": window_contract["evidence_class"],
            "confirmatory_claim_permitted": False,
            "expected_locked_periods_per_window": 24,
            "minimum_complete_periods_per_window": 20,
            "primary_sample_scope": "complete_periods",
            "primary_inference_scope": "single_window",
            "periods_per_year": 12,
            "annualization_convention": "actual_elapsed_years_v1",
            "initial_wealth": 100000.0,
            "net_exposure": float(benchmark_contract["net_exposure"]),
            "gross_leverage": float(benchmark_contract["gross_leverage"]),
            "max_long_weight": float(benchmark_contract["max_long_weight"]),
            "max_short_weight": float(benchmark_contract["max_short_weight"]),
            "turnover_convention": "drifted_pretrade_v1",
            "turnover_cost": float(benchmark_contract["turnover_cost"]),
            "financing_proration": "actual_calendar_days_v1",
            "day_count_basis": int(benchmark_contract["day_count_basis"]),
            "annual_short_borrow_rate": float(
                benchmark_contract["annual_short_borrow_rate"]),
            "annual_cash_borrow_rate": float(
                benchmark_contract["annual_cash_borrow_rate"]),
            "annual_risk_free_rate": 0.0,
            "crra_gamma": float(benchmark_contract["crra_gamma"]),
            "primary_benchmark_id": "equal_weight",
            "primary_strategy_id": "td3_ensemble10",
            "primary_superiority_test":
                "one_sided_paired_moving_block_bootstrap_crra",
            "primary_superiority_alpha": 0.05,
            "secondary_multiplicity_control":
                "holm_within_primary_vs_alternative_family",
            "bootstrap_replications": 9999,
            "bootstrap_block_length": 3,
            "inference_seed": 20261001,
            "weight_tolerance": float(benchmark_contract["weight_tolerance"]),
            "require_weight_log_hashes": True,
            "require_checkpoint_hash_for_trained_models": True,
            "require_code_and_config_hashes": True,
            "transaction_cost_sensitivity_bps": [0, 10, 25, 50],
            "annual_short_borrow_sensitivity_percent": [0, 3, 6, 10],
            "ensemble_size_sensitivity_sizes": [1, 2, 3, 5, 10],
            "figure_strategy_ids": [
                "equal_weight", "risk_parity", "shrinkage_mean_variance",
                "mean_cvar", "dynamic_nn_vine", "td3_ensemble10",
                "sac_ensemble10", "ppo_ensemble10"
            ],
            "predeclared_ensembles": [
                {"strategy_id": f"{algorithm}_ensemble10",
                 "label": f"{label} ensemble (10 seeds)", "method": label,
                 "ensemble_group": f"{algorithm}_external_gate_pass",
                 "minimum_members": 10, "include_main": True,
                 "include_inference": True}
                for algorithm, label in ALGORITHM_LABELS.items()
            ],
        }
        contract_path = temporary / "evaluation_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")

        manifest_rows: list[dict[str, Any]] = []
        for method in BENCHMARK_LABELS:
            path = benchmark_paths[method]
            manifest_rows.append({
                "strategy_id": method, "label": BENCHMARK_LABELS[method],
                "method": BENCHMARK_LABELS[method], "seed": "",
                "role": "benchmark", "completed": True, "gate_pass": True,
                "ensemble_group": "", "include_main": True,
                "include_inference": True, "report_seed_distribution": False,
                "weight_log_path": relative(path, final_root),
                "weight_log_sha256": sha256(path),
                "checkpoint_path": "not_applicable",
                "checkpoint_sha256": "not_applicable", "config_sha256": "",
                "code_sha256": "", "train_seconds": "",
                "evaluation_seconds": "", "notes": "causal fail-closed benchmark",
            })
        for key in sorted(policy_by_key):
            algorithm, seed = key
            checkpoint_row = checkpoint_by_key[key]
            checkpoint = Path(checkpoint_row["checkpoint"]).resolve()
            require(checkpoint.is_file() and
                    sha256(checkpoint) == checkpoint_row["checkpoint_sha256"],
                    f"Checkpoint changed after audit: {checkpoint}")
            label = ALGORITHM_LABELS[algorithm]
            manifest_rows.append({
                "strategy_id": f"{algorithm}_seed_{seed}",
                "label": f"{label} seed {seed}", "method": label,
                "seed": seed, "role": "proposed" if algorithm == "td3" else "ablation",
                "completed": True, "gate_pass": True,
                "ensemble_group": f"{algorithm}_external_gate_pass",
                "include_main": False, "include_inference": False,
                "report_seed_distribution": True,
                "weight_log_path": relative(policy_paths[key], final_root),
                "weight_log_sha256": sha256(policy_paths[key]),
                "checkpoint_path": relative(checkpoint, final_root),
                "checkpoint_sha256": sha256(checkpoint),
                "config_sha256": sha256(window_contract_root / "CONTENTS.sha256"),
                "code_sha256": release["release_contents_sha256"],
                "train_seconds": checkpoint_row.get("duration_seconds", ""),
                "evaluation_seconds": "", "notes": "matched external-window seed",
            })
        strategy_path = temporary / "strategy_manifest.csv"
        with strategy_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
            writer.writeheader(); writer.writerows(manifest_rows)
        input_rows = []
        for role, path in [
            ("extension_release", release_root / "CONTENTS.sha256"),
            ("window_contract", window_contract_root / "CONTENTS.sha256"),
            ("checkpoint_audit", audit_manifest_path),
            ("policy_inventory", policy_inventory_path),
            ("realized_panel", realized_panel_path),
            ("benchmark_contract", benchmark_contract_path),
            ("core_benchmark_manifest", core_benchmark_root / "benchmark_manifest.csv"),
            ("core_solver_audit", core_benchmark_root / "solver_audit.csv"),
            ("extended_benchmark_manifest", extended_benchmark_root / "benchmark_manifest.csv"),
            ("extended_solver_audit", extended_benchmark_root / "solver_audit.csv"),
        ]:
            input_rows.append({"role": role, "path": str(path.resolve()),
                               "sha256": sha256(path.resolve())})
        input_path = temporary / "input_inventory.csv"
        with input_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(input_rows[0]))
            writer.writeheader(); writer.writerows(input_rows)
        for source, name in [
            (core_benchmark_root / "benchmark_manifest.csv",
             "core_benchmark_manifest_snapshot.csv"),
            (core_benchmark_root / "solver_audit.csv",
             "core_solver_audit_snapshot.csv"),
            (extended_benchmark_root / "benchmark_manifest.csv",
             "extended_benchmark_manifest_snapshot.csv"),
            (extended_benchmark_root / "solver_audit.csv",
             "extended_solver_audit_snapshot.csv"),
            (checkpoint_audit_root / "checkpoint_audit.csv",
             "checkpoint_audit_snapshot.csv"),
            (benchmark_contract_path, "benchmark_contract_snapshot.json"),
        ]:
            shutil.copy2(source, temporary / name)
        manifest = {
            "schema_version": 1,
            "release_status": "frozen_external_development_evaluation_contract",
            "window_id": window_contract["window_id"],
            "evidence_class": window_contract["evidence_class"],
            "benchmark_count": len(BENCHMARK_LABELS),
            "individual_policy_count": 50, "ensemble_count": 5,
            "strategy_manifest_sha256": sha256(strategy_path),
            "evaluation_contract_sha256": sha256(contract_path),
            "realized_panel_sha256": sha256(realized_panel_path),
            "confirmatory_claim_permitted": False,
            "scientific_note": (
                "The original seven-asset checkpoints are not reused. All methods "
                "share the same realized panel, constraints, costs and dates."),
        }
        (temporary / "window_evaluation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (temporary / "READ_ONLY_EVALUATION_CONTRACT.txt").write_text(
            "Development/external-validity evaluation only. Do not edit.\n",
            encoding="utf-8")
        checksum = [f"{sha256(path)}  {path.name}" for path in
                    sorted(temporary.iterdir()) if path.is_file() and
                    path.name != "CONTENTS.sha256"]
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(checksum) + "\n", encoding="ascii")
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--window-contract", required=True, type=Path)
    parser.add_argument("--checkpoint-audit", required=True, type=Path)
    parser.add_argument("--policy-inventory", required=True, type=Path)
    parser.add_argument("--core-benchmarks", required=True, type=Path)
    parser.add_argument("--extended-benchmarks", required=True, type=Path)
    parser.add_argument("--realized-panel", required=True, type=Path)
    parser.add_argument("--benchmark-contract", type=Path, default=Path(
        "publication_pipeline_draft/config/benchmark_contract_v2.json"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = materialize(
            args.repo_root, args.release.resolve(), args.window_contract.resolve(),
            args.checkpoint_audit.resolve(), args.policy_inventory.resolve(),
            args.core_benchmarks.resolve(), args.extended_benchmarks.resolve(),
            args.realized_panel.resolve(), args.benchmark_contract.resolve(),
            args.output)
    except (OSError, ValueError, json.JSONDecodeError, WindowEvaluationError,
            ExtensionReleaseError) as error:
        print(f"WINDOW EVALUATION CONTRACT FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
