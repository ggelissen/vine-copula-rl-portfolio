#!/usr/bin/env python3
"""Build focused seed ensembles and score one window with common accounting."""

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

import pandas as pd

from publication_pipeline_draft.focused_window_training_protocol import (
    validate_protocol,
)
from publication_pipeline_draft.publication_pipeline import (
    Contract, ProtocolError, read_and_validate_weights, read_realized_panel,
    score_strategy,
)
from publication_pipeline_draft.window_evaluation_protocol import (
    verify_benchmark_family,
)


class FocusedScoreError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FocusedScoreError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def score(protocol_path: Path, inventory_path: Path, audit_root: Path,
          realized_path: Path, benchmark_root: Path,
          benchmark_contract_path: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"Output already exists: {output}")
    protocol, protocol_sha256 = validate_protocol(protocol_path)
    inventory = read_csv(inventory_path)
    audit_manifest_path = audit_root / "focused_sweep_audit_manifest.json"
    audit_table_path = audit_root / "focused_checkpoint_audit.csv"
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    audit_rows = read_csv(audit_table_path)
    require(audit_manifest.get("status") == "focused_window_sweep_audit_passed" and
            len(inventory) == len(audit_rows) == 15,
            "Exact focused policy inventory and audit are required.")
    audit_by_key = {(row["experiment_id"], int(row["seed"])): row
                    for row in audit_rows}
    inventory_by_key = {(row["experiment_id"], int(row["seed"])): row
                        for row in inventory}
    require(set(audit_by_key) == set(inventory_by_key),
            "Focused policy inventory and checkpoint audit differ.")
    windows = {row["window_id"] for row in inventory}
    require(len(windows) == 1, "Focused score input must contain one window.")
    window_id = next(iter(windows))
    benchmark_contract = json.loads(
        benchmark_contract_path.read_text(encoding="utf-8"))
    benchmark_ids = set(protocol["financial_benchmarks"])
    benchmark_paths = verify_benchmark_family(
        benchmark_root, benchmark_ids,
        {int(value) for value in
         benchmark_contract["optimizer_allowed_convergence_codes"]},
        max(float(benchmark_contract["weight_tolerance"]), 1e-8))
    for key, row in inventory_by_key.items():
        path = Path(row["weight_file"])
        checkpoint = Path(row["checkpoint"])
        require(path.is_file() and sha256(path) == row["sha256"],
                f"Focused weight file changed: {key}")
        require(checkpoint.is_file() and
                sha256(checkpoint) == row["checkpoint_sha256"] ==
                audit_by_key[key]["checkpoint_sha256"],
                f"Focused checkpoint changed: {key}")

    contract_value = {
        "schema_version": 1,
        "evaluation_id": f"focused_retrospective_{window_id}",
        "expected_locked_periods_per_window": 24,
        "minimum_complete_periods_per_window": 20,
        "primary_sample_scope": "complete_periods",
        "periods_per_year": 12,
        "annualization_convention": "actual_elapsed_years_v1",
        "initial_wealth": 100000.0,
        "net_exposure": 1.0, "gross_leverage": 1.5,
        "max_long_weight": 0.6, "max_short_weight": 0.2,
        "turnover_convention": "drifted_pretrade_v1",
        "turnover_cost": 0.001,
        "financing_proration": "actual_calendar_days_v1",
        "day_count_basis": 365,
        "annual_short_borrow_rate": 0.03,
        "annual_cash_borrow_rate": 0.02,
        "annual_risk_free_rate": 0.0,
        "crra_gamma": float(protocol["crra_gamma"]),
        "primary_benchmark_id": "zero_vine_features_keep_cvar_observation_ensemble5",
        "primary_strategy_id": "full_vine_state_and_cvar_observation_ensemble5",
        "primary_superiority_test":
            "one_sided_paired_moving_block_bootstrap_crra",
        "primary_superiority_alpha": 0.05,
        "secondary_multiplicity_control":
            "holm_within_primary_vs_alternative_family",
        "bootstrap_replications": int(protocol["bootstrap"]["replications"]),
        "bootstrap_block_length": int(protocol["bootstrap"]["block_length_periods"]),
        "inference_seed": int(protocol["bootstrap"]["seed"]),
        "weight_tolerance": 0.000001,
        "predeclared_ensembles": [],
        "require_weight_log_hashes": True,
        "require_checkpoint_hash_for_trained_models": False,
        "require_code_and_config_hashes": False,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        contract_path = temporary / "focused_evaluation_contract.json"
        contract_path.write_text(
            json.dumps(contract_value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        contract = Contract.read(contract_path)
        realized, assets = read_realized_panel(realized_path, contract)
        weight_columns = [f"w_{asset}" for asset in assets]
        strategy_rows: list[pd.DataFrame] = []
        seed_weights: dict[str, list[pd.DataFrame]] = {}

        for key in sorted(inventory_by_key):
            experiment, seed = key
            item = inventory_by_key[key]
            strategy_id = f"{experiment}_seed_{seed}"
            row = pd.Series({
                "strategy_id": strategy_id, "role": "ablation",
                "weight_log_path": str(Path(item["weight_file"]).resolve()),
                "weight_log_sha256": item["sha256"],
            })
            # Use a temporary manifest location-neutral path by passing an
            # absolute weight path relative to the temporary directory below.
            relative_path = Path(os.path.relpath(
                Path(item["weight_file"]).resolve(), temporary.resolve()))
            row["weight_log_path"] = relative_path.as_posix()
            weight, digest = read_and_validate_weights(
                row, temporary / "focused_strategy_manifest.csv",
                realized, assets, contract)
            seed_weights.setdefault(experiment, []).append(weight)
            scored = score_strategy(strategy_id, weight, realized, assets, contract)
            scored["experiment_id"] = experiment
            scored["strategy_level"] = "seed"
            scored["seed"] = seed
            strategy_rows.append(scored)

        for experiment, members in sorted(seed_weights.items()):
            require(len(members) == 5,
                    f"Focused ensemble {experiment} does not have five seeds.")
            ensemble = members[0][["window_id", "decision_date",
                                   "holding_end_date"]].copy()
            for name in weight_columns:
                ensemble[name] = sum(member[name].to_numpy(float)
                                     for member in members) / len(members)
            strategy_id = f"{experiment}_ensemble5"
            scored = score_strategy(strategy_id, ensemble, realized, assets, contract)
            scored["experiment_id"] = experiment
            scored["strategy_level"] = "ensemble"
            scored["seed"] = ""
            strategy_rows.append(scored)
        for benchmark_id in protocol["financial_benchmarks"]:
            path = benchmark_paths[benchmark_id]
            row = pd.Series({
                "strategy_id": benchmark_id, "role": "benchmark",
                "weight_log_path": Path(os.path.relpath(
                    path.resolve(), temporary.resolve())).as_posix(),
                "weight_log_sha256": sha256(path),
            })
            weight, _ = read_and_validate_weights(
                row, temporary / "focused_strategy_manifest.csv",
                realized, assets, contract)
            scored = score_strategy(
                benchmark_id, weight, realized, assets, contract)
            scored["experiment_id"] = benchmark_id
            scored["strategy_level"] = "benchmark"
            scored["seed"] = ""
            strategy_rows.append(scored)
        panel = pd.concat(strategy_rows, ignore_index=True)
        require(len(panel) == 24 * 24 and panel["strategy_id"].nunique() == 24,
                "Focused common accounting did not create 15 seeds, 3 ensembles, and 6 benchmarks.")
        panel.to_csv(temporary / "focused_scored_period_panel.csv", index=False,
                     date_format="%Y-%m-%d")
        pd.DataFrame(inventory).to_csv(
            temporary / "focused_policy_inventory_snapshot.csv", index=False)
        manifest = {
            "schema_version": 1,
            "status": "focused_window_common_accounting_complete",
            "window_id": window_id,
            "protocol_sha256": protocol_sha256,
            "realized_panel_sha256": sha256(realized_path),
            "policy_inventory_sha256": sha256(inventory_path),
            "checkpoint_audit_sha256": sha256(audit_table_path),
            "seed_strategy_count": 15, "ensemble_strategy_count": 3,
            "benchmark_strategy_count": 6,
            "strategy_count": 24, "periods_per_strategy": 24,
            "common_realized_returns": True,
            "common_drifted_cost_accounting": True,
            "confirmatory_claim_permitted": False,
        }
        (temporary / "focused_score_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
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
    parser.add_argument("--protocol", type=Path, default=Path(
        "publication_pipeline_draft/config/focused_walk_forward_mechanisms_v1.json"))
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--realized", required=True, type=Path)
    parser.add_argument("--benchmarks", required=True, type=Path)
    parser.add_argument("--benchmark-contract", type=Path, default=Path(
        "publication_pipeline_draft/config/benchmark_contract_v3.json"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = score(args.protocol.resolve(), args.inventory.resolve(),
                       args.audit.resolve(), args.realized.resolve(),
                       args.benchmarks.resolve(),
                       args.benchmark_contract.resolve(), args.output)
    except (OSError, ValueError, KeyError, ProtocolError,
            FocusedScoreError) as error:
        print(f"FOCUSED WINDOW SCORE FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
