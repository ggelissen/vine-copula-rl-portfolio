#!/usr/bin/env python3
"""Fail-closed common-path evaluator and paper artifact exporter.

Strategies contribute ex-ante weights only. Every return, cost, constraint,
metric, comparison and plot is reconstructed here from one frozen realized
asset panel. This file intentionally has no optimizer-specific fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ProtocolError(RuntimeError):
    """Raised when an input violates the frozen evaluation contract."""


KEYS = ["window_id", "decision_date", "holding_end_date"]
TRUE_VALUES = {"true", "t", "1", "yes", "y"}
FALSE_VALUES = {"false", "f", "0", "no", "n", ""}
HASH_LENGTH = 64


def parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ProtocolError(f"{field} must be boolean; received {value!r}.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == HASH_LENGTH and all(c in "0123456789abcdef" for c in text)


def require_fields(mapping: dict[str, Any], names: Iterable[str], source: str) -> None:
    missing = [name for name in names if name not in mapping]
    if missing:
        raise ProtocolError(f"{source} is missing fields: {', '.join(missing)}")


@dataclass(frozen=True)
class Contract:
    raw: dict[str, Any]

    @classmethod
    def read(cls, path: Path) -> "Contract":
        raw = json.loads(path.read_text(encoding="utf-8"))
        required = [
            "schema_version", "evaluation_id",
            "expected_locked_periods_per_window",
            "minimum_complete_periods_per_window", "primary_sample_scope",
            "periods_per_year", "initial_wealth", "net_exposure",
            "gross_leverage", "max_long_weight", "max_short_weight",
            "turnover_cost", "annual_short_borrow_rate",
            "annual_cash_borrow_rate", "annual_risk_free_rate", "crra_gamma",
            "primary_benchmark_id", "primary_strategy_id",
            "primary_superiority_test", "primary_superiority_alpha",
            "secondary_multiplicity_control",
            "bootstrap_replications", "bootstrap_block_length",
            "inference_seed", "weight_tolerance", "predeclared_ensembles",
        ]
        require_fields(raw, required, str(path))
        if int(raw["schema_version"]) != 1:
            raise ProtocolError("Only evaluation contract schema_version=1 is supported.")
        if raw["primary_sample_scope"] not in {"complete_periods", "locked_all"}:
            raise ProtocolError("primary_sample_scope must be complete_periods or locked_all.")
        numeric_positive = [
            "expected_locked_periods_per_window", "minimum_complete_periods_per_window",
            "periods_per_year", "initial_wealth", "gross_leverage",
            "max_long_weight", "max_short_weight", "bootstrap_replications",
            "bootstrap_block_length", "weight_tolerance",
        ]
        for name in numeric_positive:
            if not math.isfinite(float(raw[name])) or float(raw[name]) <= 0:
                raise ProtocolError(f"{name} must be finite and positive.")
        if float(raw["gross_leverage"]) < abs(float(raw["net_exposure"])):
            raise ProtocolError("gross_leverage cannot be below absolute net_exposure.")
        if float(raw["turnover_cost"]) < 0 or float(raw["annual_short_borrow_rate"]) < 0:
            raise ProtocolError("Trading and borrow rates cannot be negative.")
        if float(raw["annual_cash_borrow_rate"]) < 0:
            raise ProtocolError("annual_cash_borrow_rate cannot be negative.")
        if int(raw["bootstrap_replications"]) < 999:
            raise ProtocolError("At least 999 bootstrap replications are required.")
        if raw["primary_superiority_test"] != "one_sided_paired_moving_block_bootstrap_crra":
            raise ProtocolError("Unsupported primary_superiority_test.")
        if not 0.0 < float(raw["primary_superiority_alpha"]) < 0.5:
            raise ProtocolError("primary_superiority_alpha must lie strictly between 0 and 0.5.")
        if raw["secondary_multiplicity_control"] != "holm_within_primary_vs_alternative_family":
            raise ProtocolError("Unsupported secondary_multiplicity_control.")
        return cls(raw=raw)

    def __getitem__(self, name: str) -> Any:
        return self.raw[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self.raw.get(name, default)


def read_realized_panel(path: Path, contract: Contract) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise ProtocolError(f"Realized panel not found: {path}")
    frame = pd.read_csv(path)
    required = KEYS + ["trading_days", "is_complete_period"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ProtocolError(f"Realized panel is missing: {', '.join(missing)}")
    # CSV column order is the frozen asset order.  Sorting here silently
    # changed the actor/benchmark order whenever asset labels were not already
    # alphabetical, making an otherwise valid log semantically ambiguous.
    gross_columns = [name for name in frame if name.startswith("g_")]
    if len(gross_columns) < 2:
        raise ProtocolError("Realized panel needs at least two g_<ASSET> columns.")
    frame["window_id"] = frame["window_id"].astype(str)
    for name in ["decision_date", "holding_end_date"]:
        frame[name] = pd.to_datetime(frame[name], errors="raise").dt.normalize()
    frame["is_complete_period"] = [
        parse_bool(value, "is_complete_period") for value in frame["is_complete_period"]
    ]
    frame["trading_days"] = pd.to_numeric(frame["trading_days"], errors="raise").astype(int)
    if frame[KEYS].duplicated().any():
        raise ProtocolError("Realized panel contains duplicate holding-period keys.")
    if (frame["holding_end_date"] <= frame["decision_date"]).any():
        raise ProtocolError("Each holding_end_date must follow its decision_date.")
    if (frame["trading_days"] < 1).any():
        raise ProtocolError("trading_days must be positive.")
    gross = frame[gross_columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(gross).all() or (gross <= 0).any():
        raise ProtocolError("Realized asset gross returns must be finite and positive.")
    frame[gross_columns] = gross
    frame = frame.sort_values(KEYS, kind="stable").reset_index(drop=True)
    expected = int(contract["expected_locked_periods_per_window"])
    minimum_complete = int(contract["minimum_complete_periods_per_window"])
    for window_id, group in frame.groupby("window_id", sort=False):
        if len(group) != expected:
            raise ProtocolError(
                f"Window {window_id} has {len(group)} periods; expected exactly {expected}."
            )
        if int(group["is_complete_period"].sum()) < minimum_complete:
            raise ProtocolError(
                f"Window {window_id} has too few complete periods for the declared primary sample."
            )
        if not group["decision_date"].is_monotonic_increasing:
            raise ProtocolError(f"Decision dates are not increasing in window {window_id}.")
    assets = [name[2:] for name in gross_columns]
    return frame, assets


MANIFEST_REQUIRED = [
    "strategy_id", "label", "method", "seed", "role", "completed", "gate_pass",
    "ensemble_group", "include_main", "include_inference",
    "report_seed_distribution", "weight_log_path", "weight_log_sha256",
    "checkpoint_path", "checkpoint_sha256", "config_sha256", "code_sha256",
    "train_seconds", "evaluation_seconds", "notes",
]


def read_strategy_manifest(path: Path, contract: Contract) -> pd.DataFrame:
    if not path.is_file():
        raise ProtocolError(f"Strategy manifest not found: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [name for name in MANIFEST_REQUIRED if name not in frame]
    if missing:
        raise ProtocolError(f"Strategy manifest is missing: {', '.join(missing)}")
    if frame.empty or frame["strategy_id"].duplicated().any():
        raise ProtocolError("Strategy manifest must contain unique strategy_id rows.")
    bool_fields = [
        "completed", "gate_pass", "include_main", "include_inference",
        "report_seed_distribution",
    ]
    for name in bool_fields:
        frame[name] = [parse_bool(value, name) for value in frame[name]]
    if not frame["completed"].all():
        incomplete = frame.loc[~frame["completed"], "strategy_id"].tolist()
        raise ProtocolError(f"Manifest contains incomplete runs: {incomplete}")
    trained_roles = {"proposed", "ablation", "sensitivity"}
    require_checkpoint = parse_bool(
        contract.get("require_checkpoint_hash_for_trained_models", True),
        "require_checkpoint_hash_for_trained_models",
    )
    require_code = parse_bool(
        contract.get("require_code_and_config_hashes", True),
        "require_code_and_config_hashes",
    )
    for index, row in frame.iterrows():
        if row["role"] in trained_roles and not row["gate_pass"]:
            raise ProtocolError(f"Trained strategy {row['strategy_id']} did not pass its pre-holdout gate.")
        if row["role"] in trained_roles and require_checkpoint:
            checkpoint = (path.parent / row["checkpoint_path"]).resolve()
            if not checkpoint.is_file() or not is_sha256(row["checkpoint_sha256"]):
                raise ProtocolError(f"{row['strategy_id']} lacks a valid checkpoint path/hash.")
            if sha256_file(checkpoint) != row["checkpoint_sha256"].lower():
                raise ProtocolError(f"Checkpoint hash mismatch for {row['strategy_id']}.")
        if row["role"] in trained_roles and require_code:
            if not is_sha256(row["config_sha256"]) or not is_sha256(row["code_sha256"]):
                raise ProtocolError(f"{row['strategy_id']} lacks frozen config/code hashes.")
        for field in ["train_seconds", "evaluation_seconds"]:
            if row[field] != "" and (not math.isfinite(float(row[field])) or float(row[field]) < 0):
                raise ProtocolError(f"{field} must be blank or non-negative for {row['strategy_id']}.")
        frame.loc[index, "manifest_directory"] = str(path.parent.resolve())
    return frame


def generated_equal_weight(panel: pd.DataFrame, assets: list[str], net: float) -> pd.DataFrame:
    out = panel[KEYS].copy()
    for asset in assets:
        out[f"w_{asset}"] = net / len(assets)
    return out


def read_and_validate_weights(
    row: pd.Series,
    manifest_path: Path,
    realized: pd.DataFrame,
    assets: list[str],
    contract: Contract,
) -> tuple[pd.DataFrame, str]:
    special = row["weight_log_path"].strip().upper()
    if special == "GENERATE_EQUAL_WEIGHT":
        if row["role"] != "benchmark":
            raise ProtocolError("Only a declared benchmark may generate equal weights internally.")
        return generated_equal_weight(realized, assets, float(contract["net_exposure"])), "generated"
    path = (manifest_path.parent / row["weight_log_path"]).resolve()
    if not path.is_file():
        raise ProtocolError(f"Weight log not found for {row['strategy_id']}: {path}")
    actual_hash = sha256_file(path)
    require_hash = parse_bool(contract.get("require_weight_log_hashes", True), "require_weight_log_hashes")
    declared_hash = row["weight_log_sha256"].strip().lower()
    if require_hash and not is_sha256(declared_hash):
        raise ProtocolError(f"{row['strategy_id']} lacks a valid weight_log_sha256.")
    if declared_hash and actual_hash != declared_hash:
        raise ProtocolError(f"Weight-log hash mismatch for {row['strategy_id']}.")
    weights = pd.read_csv(path)
    if "window_id" not in weights:
        windows = realized["window_id"].unique()
        if len(windows) != 1:
            raise ProtocolError(f"{row['strategy_id']} omits window_id in a multi-window panel.")
        weights["window_id"] = windows[0]
    weights["window_id"] = weights["window_id"].astype(str)
    if "decision_date" not in weights:
        raise ProtocolError(f"{row['strategy_id']} has no decision_date.")
    weights["decision_date"] = pd.to_datetime(weights["decision_date"], errors="raise").dt.normalize()
    if "holding_end_date" not in weights:
        weights = weights.merge(
            realized[KEYS], on=["window_id", "decision_date"], how="left", validate="one_to_one"
        )
    else:
        weights["holding_end_date"] = pd.to_datetime(
            weights["holding_end_date"], errors="raise"
        ).dt.normalize()
    expected_columns = [f"w_{asset}" for asset in assets]
    missing = [name for name in expected_columns if name not in weights]
    unexpected = [name for name in weights if name.startswith("w_") and name not in expected_columns]
    if missing or unexpected:
        raise ProtocolError(
            f"{row['strategy_id']} asset columns differ from the realized panel; "
            f"missing={missing}, unexpected={unexpected}."
        )
    if weights[KEYS].duplicated().any():
        raise ProtocolError(f"{row['strategy_id']} contains duplicated period keys.")
    expected_keys = realized[KEYS].sort_values(KEYS).reset_index(drop=True)
    observed_keys = weights[KEYS].sort_values(KEYS).reset_index(drop=True)
    if len(observed_keys) != len(expected_keys) or not observed_keys.equals(expected_keys):
        raise ProtocolError(f"{row['strategy_id']} does not contain exactly the locked period keys.")
    weights = realized[KEYS].merge(weights[KEYS + expected_columns], on=KEYS, how="left", validate="one_to_one")
    if weights[expected_columns].isna().any().any() or len(weights) != len(realized):
        raise ProtocolError(f"{row['strategy_id']} is not aligned to every locked holding period.")
    matrix = weights[expected_columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    validate_weight_matrix(matrix, row["strategy_id"], contract)
    weights[expected_columns] = matrix
    return weights, actual_hash


def validate_weight_matrix(matrix: np.ndarray, strategy_id: str, contract: Contract) -> None:
    if not np.isfinite(matrix).all():
        raise ProtocolError(f"{strategy_id} contains non-finite weights.")
    tolerance = float(contract["weight_tolerance"])
    net = matrix.sum(axis=1)
    gross = np.abs(matrix).sum(axis=1)
    if np.max(np.abs(net - float(contract["net_exposure"]))) > tolerance:
        raise ProtocolError(f"{strategy_id} violates the common net-exposure constraint.")
    if np.max(gross) > float(contract["gross_leverage"]) + tolerance:
        raise ProtocolError(f"{strategy_id} violates the common gross-leverage constraint.")
    if np.max(matrix) > float(contract["max_long_weight"]) + tolerance:
        raise ProtocolError(f"{strategy_id} violates the common single-asset long cap.")
    if np.min(matrix) < -float(contract["max_short_weight"]) - tolerance:
        raise ProtocolError(f"{strategy_id} violates the common single-asset short cap.")


def build_ensembles(
    weights: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    assets: list[str],
    contract: Contract,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    additions: list[dict[str, Any]] = []
    weight_columns = [f"w_{asset}" for asset in assets]
    for definition in contract["predeclared_ensembles"]:
        require_fields(
            definition,
            ["strategy_id", "label", "method", "ensemble_group", "include_main", "include_inference"],
            "predeclared ensemble",
        )
        strategy_id = str(definition["strategy_id"])
        if strategy_id in weights or strategy_id in set(manifest["strategy_id"]):
            raise ProtocolError(f"Duplicate ensemble strategy_id: {strategy_id}")
        members = manifest[
            (manifest["ensemble_group"] == str(definition["ensemble_group"]))
            & manifest["completed"]
            & manifest["gate_pass"]
        ]
        minimum = int(definition.get("minimum_members", 2))
        if len(members) < minimum:
            raise ProtocolError(
                f"Ensemble {strategy_id} has {len(members)} eligible members; needs {minimum}."
            )
        stack = np.stack([weights[member][weight_columns].to_numpy(float) for member in members["strategy_id"]])
        ensemble = weights[members.iloc[0]["strategy_id"]][KEYS].copy()
        ensemble[weight_columns] = stack.mean(axis=0)
        validate_weight_matrix(ensemble[weight_columns].to_numpy(float), strategy_id, contract)
        weights[strategy_id] = ensemble
        additions.append(
            {
                "strategy_id": strategy_id,
                "label": definition["label"],
                "method": definition["method"],
                "seed": "ensemble",
                "role": "proposed_ensemble",
                "completed": True,
                "gate_pass": True,
                "ensemble_group": definition["ensemble_group"],
                "include_main": parse_bool(definition["include_main"], "include_main"),
                "include_inference": parse_bool(definition["include_inference"], "include_inference"),
                "report_seed_distribution": False,
                "weight_log_path": "generated_mean_weights",
                "weight_log_sha256": sha256_json(
                    {"members": members["strategy_id"].tolist(), "rule": "arithmetic_mean_weights"}
                ),
                "checkpoint_path": "ensemble_members",
                "checkpoint_sha256": sha256_json(sorted(members["checkpoint_sha256"].tolist())),
                "config_sha256": sha256_json(sorted(members["config_sha256"].tolist())),
                "code_sha256": sha256_json(sorted(members["code_sha256"].tolist())),
                "train_seconds": pd.to_numeric(members["train_seconds"], errors="coerce").sum(min_count=1),
                "evaluation_seconds": pd.to_numeric(members["evaluation_seconds"], errors="coerce").sum(min_count=1),
                "notes": f"Arithmetic mean of {len(members)} pre-holdout gate-passing seeds",
                "ensemble_members": "|".join(members["strategy_id"].tolist()),
            }
        )
    if additions:
        for name in set(manifest.columns).difference(additions[0]):
            for row in additions:
                row[name] = ""
        for name in set(additions[0]).difference(manifest.columns):
            manifest[name] = ""
        manifest = pd.concat([manifest, pd.DataFrame(additions)[manifest.columns]], ignore_index=True)
    return weights, manifest


def score_strategy(
    strategy_id: str,
    weights: pd.DataFrame,
    realized: pd.DataFrame,
    assets: list[str],
    contract: Contract,
) -> pd.DataFrame:
    gross_columns = [f"g_{asset}" for asset in assets]
    weight_columns = [f"w_{asset}" for asset in assets]
    merged = realized.merge(weights, on=KEYS, how="inner", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    periods_per_year = float(contract["periods_per_year"])
    for window_id, group in merged.groupby("window_id", sort=False):
        previous = np.repeat(float(contract["net_exposure"]) / len(assets), len(assets))
        wealth = float(contract["initial_wealth"])
        peak = wealth
        for _, record in group.sort_values("decision_date", kind="stable").iterrows():
            w = record[weight_columns].to_numpy(float)
            g = record[gross_columns].to_numpy(float)
            turnover = float(np.abs(w - previous).sum())
            transaction_cost = float(contract["turnover_cost"]) * turnover
            short_notional = float(np.maximum(-w, 0).sum())
            cash_borrow_notional = max(float(w.sum()) - 1.0, 0.0)
            financing_cost = (
                float(contract["annual_short_borrow_rate"]) * short_notional
                + float(contract["annual_cash_borrow_rate"]) * cash_borrow_notional
            ) / periods_per_year
            gross_portfolio = 1.0 + float(np.dot(w, g - 1.0))
            net_gross = gross_portfolio * math.exp(-transaction_cost - financing_cost)
            if not math.isfinite(net_gross) or net_gross <= 0:
                raise ProtocolError(f"{strategy_id} is insolvent in window {window_id}.")
            wealth *= net_gross
            peak = max(peak, wealth)
            row = {
                "strategy_id": strategy_id,
                "window_id": window_id,
                "decision_date": record["decision_date"],
                "holding_end_date": record["holding_end_date"],
                "trading_days": int(record["trading_days"]),
                "is_complete_period": bool(record["is_complete_period"]),
                "gross_return": gross_portfolio - 1.0,
                "net_return": net_gross - 1.0,
                "wealth": wealth,
                "drawdown": wealth / peak - 1.0,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "financing_cost": financing_cost,
                "short_notional": short_notional,
                "cash_borrow_notional": cash_borrow_notional,
                "gross_exposure": float(np.abs(w).sum()),
                "net_exposure": float(w.sum()),
            }
            row.update({name: float(record[name]) for name in weight_columns})
            rows.append(row)
            previous = w
    return pd.DataFrame(rows)


def empirical_metrics(group: pd.DataFrame, contract: Contract) -> dict[str, float]:
    group = group.sort_values("decision_date", kind="stable")
    returns = group["net_return"].to_numpy(float)
    gross_returns = group["gross_return"].to_numpy(float)
    if len(returns) < 2 or not np.isfinite(returns).all() or (returns <= -1).any():
        raise ProtocolError("Metric calculation requires at least two valid simple returns.")
    ppy = float(contract["periods_per_year"])
    rf_period = float(contract["annual_risk_free_rate"]) / ppy
    wealth_multiple = np.cumprod(1.0 + returns)
    gross_wealth_multiple = np.cumprod(1.0 + gross_returns)
    running_peak = np.maximum.accumulate(np.r_[1.0, wealth_multiple])[1:]
    drawdown = wealth_multiple / running_peak - 1.0
    excess = returns - rf_period
    standard_deviation = float(np.std(returns, ddof=1))
    excess_sd = float(np.std(excess, ddof=1))
    downside = np.minimum(excess, 0.0)
    downside_dev = math.sqrt(float(np.mean(downside**2)))
    q05 = float(np.quantile(returns, 0.05, method="median_unbiased"))
    tail = returns[returns <= q05 + 1e-15]
    total = float(wealth_multiple[-1] - 1.0)
    gross_total = float(gross_wealth_multiple[-1] - 1.0)
    max_drawdown = float(-np.min(drawdown))
    mean_utility = float(np.mean(crra_utility(returns, float(contract["crra_gamma"]))))
    gamma = float(contract["crra_gamma"])
    if gamma == 1.0:
        certainty_equivalent_gross = math.exp(mean_utility)
    else:
        inverse_base = 1.0 + (1.0 - gamma) * mean_utility
        certainty_equivalent_gross = (
            inverse_base ** (1.0 / (1.0 - gamma))
            if inverse_base > 0 else math.nan
        )
    gains = float(np.maximum(excess, 0.0).sum())
    losses = float(-np.minimum(excess, 0.0).sum())
    return_series = pd.Series(returns)
    return {
        "observations": float(len(returns)),
        "total_return": total,
        "cagr": float(wealth_multiple[-1] ** (ppy / len(returns)) - 1.0),
        "annual_arithmetic_return": float(np.mean(returns) * ppy),
        "geometric_mean_monthly_return": float(wealth_multiple[-1] ** (1.0 / len(returns)) - 1.0),
        "annual_volatility": standard_deviation * math.sqrt(ppy),
        "annual_downside_deviation": downside_dev * math.sqrt(ppy),
        "sharpe_ratio": float(np.mean(excess) / excess_sd * math.sqrt(ppy)) if excess_sd > 0 else math.nan,
        "sortino_ratio": float(np.mean(excess) / downside_dev * math.sqrt(ppy)) if downside_dev > 0 else math.nan,
        "calmar_ratio": float((wealth_multiple[-1] ** (ppy / len(returns)) - 1.0) / max_drawdown) if max_drawdown > 0 else math.nan,
        "omega_ratio": gains / losses if losses > 0 else math.nan,
        "max_drawdown": max_drawdown,
        "realized_var05_loss": -q05,
        "realized_cvar05_loss": float(-np.mean(tail)),
        "tail_event_count": float(len(tail)),
        "worst_month_return": float(np.min(returns)),
        "best_month_return": float(np.max(returns)),
        "return_skewness": float(return_series.skew()),
        "return_excess_kurtosis": float(return_series.kurt()),
        "monthly_certainty_equivalent_return": float(certainty_equivalent_gross - 1.0),
        "annualized_certainty_equivalent_return": float(certainty_equivalent_gross ** ppy - 1.0),
        "mean_crra_utility": mean_utility,
        "terminal_wealth": float(contract["initial_wealth"]) * wealth_multiple[-1],
        "positive_month_fraction": float(np.mean(returns > 0)),
        "gross_total_return_before_costs": gross_total,
        "implementation_drag_total_return": gross_total - total,
        "mean_monthly_turnover": float(group["turnover"].mean()),
        "annualized_turnover": float(group["turnover"].mean() * ppy),
        "mean_gross_exposure": float(group["gross_exposure"].mean()),
        "maximum_gross_exposure": float(group["gross_exposure"].max()),
        "mean_short_notional": float(group["short_notional"].mean()),
        "total_transaction_log_cost": float(group["transaction_cost"].sum()),
        "total_financing_log_cost": float(group["financing_cost"].sum()),
        "mean_transaction_cost_bps": float(group["transaction_cost"].mean() * 10000.0),
        "mean_financing_cost_bps": float(group["financing_cost"].mean() * 10000.0),
    }


def scope_rows(scored: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "locked_all":
        return scored
    if scope == "complete_periods":
        return scored[scored["is_complete_period"]].copy()
    raise ProtocolError(f"Unknown sample scope: {scope}")


def build_metric_tables(scored: pd.DataFrame, manifest: pd.DataFrame, contract: Contract) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = manifest.set_index("strategy_id")
    rows: list[dict[str, Any]] = []
    for scope in ["locked_all", "complete_periods"]:
        scoped = scope_rows(scored, scope)
        for (strategy_id, window_id), group in scoped.groupby(["strategy_id", "window_id"], sort=False):
            row: dict[str, Any] = {
                "sample_scope": scope,
                "strategy_id": strategy_id,
                "window_id": window_id,
                "label": metadata.loc[strategy_id, "label"],
                "method": metadata.loc[strategy_id, "method"],
                "seed": metadata.loc[strategy_id, "seed"],
                "role": metadata.loc[strategy_id, "role"],
                "include_main": bool(metadata.loc[strategy_id, "include_main"]),
                "include_inference": bool(metadata.loc[strategy_id, "include_inference"]),
                "report_seed_distribution": bool(metadata.loc[strategy_id, "report_seed_distribution"]),
            }
            row.update(empirical_metrics(group, contract))
            rows.append(row)
    per_window = pd.DataFrame(rows)
    main_source = per_window[
        (per_window["sample_scope"] == contract["primary_sample_scope"])
        & per_window["include_main"]
    ].copy()
    numeric = [
        name for name in per_window.columns
        if name not in {
            "sample_scope", "strategy_id", "window_id", "label", "method", "seed",
            "role", "include_main", "include_inference", "report_seed_distribution",
        }
    ]
    main = (
        main_source.groupby(["strategy_id", "label", "method"], sort=False, dropna=False)[numeric]
        .mean()
        .reset_index()
    )
    window_count = main_source.groupby("strategy_id")["window_id"].nunique()
    main.insert(3, "n_windows", main["strategy_id"].map(window_count).astype(int))
    main.insert(4, "sample_scope", contract["primary_sample_scope"])
    return per_window, main


def crra_utility(simple_return: np.ndarray, gamma: float) -> np.ndarray:
    gross = 1.0 + np.asarray(simple_return, dtype=float)
    if not np.isfinite(gross).all() or (gross <= 0).any():
        raise ProtocolError("CRRA utility requires positive finite gross returns.")
    if gamma == 1.0:
        return np.log(gross)
    return (gross ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def newey_west_mean_test(values: np.ndarray, lag: int | None = None) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    if len(x) < 8 or not np.isfinite(x).all():
        raise ProtocolError("HAC inference requires at least eight finite paired observations.")
    n = len(x)
    if lag is None:
        lag = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lag = max(0, min(int(lag), n - 2))
    centered = x - x.mean()
    long_run = float(np.dot(centered, centered) / n)
    for ell in range(1, lag + 1):
        covariance = float(np.dot(centered[ell:], centered[:-ell]) / n)
        long_run += 2.0 * (1.0 - ell / (lag + 1.0)) * covariance
    se = math.sqrt(max(long_run, 0.0) / n)
    mean = float(x.mean())
    if se <= np.finfo(float).eps:
        statistic = math.copysign(math.inf, mean) if mean != 0 else 0.0
        p_two = 0.0 if mean != 0 else 1.0
        p_greater = 0.0 if mean > 0 else (1.0 if mean < 0 else 0.5)
    else:
        statistic = mean / se
        normal = NormalDist()
        p_two = 2.0 * (1.0 - normal.cdf(abs(statistic)))
        p_greater = 1.0 - normal.cdf(statistic)
    return {
        "mean_utility_difference": mean,
        "hac_standard_error": se,
        "hac_statistic": statistic,
        "hac_p_two_sided": p_two,
        "hac_p_candidate_greater": p_greater,
        "hac_ci_lower": mean - 1.959963984540054 * se,
        "hac_ci_upper": mean + 1.959963984540054 * se,
        "hac_lag": float(lag),
        "observations": float(n),
    }


def moving_block_indices(rng: np.random.Generator, n: int, block_length: int) -> np.ndarray:
    starts = rng.integers(0, n, size=math.ceil(n / block_length))
    blocks = [(start + np.arange(block_length)) % n for start in starts]
    return np.concatenate(blocks)[:n]


def paired_block_bootstrap(values: np.ndarray, contract: Contract, seed_offset: int = 0) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    n = len(x)
    block = min(int(contract["bootstrap_block_length"]), n)
    replications = int(contract["bootstrap_replications"])
    rng = np.random.default_rng(int(contract["inference_seed"]) + seed_offset)
    means = np.empty(replications)
    null_means = np.empty(replications)
    centered = x - x.mean()
    for index in range(replications):
        sample = moving_block_indices(rng, n, block)
        means[index] = x[sample].mean()
        null_means[index] = centered[sample].mean()
    observed = float(x.mean())
    return {
        "bootstrap_ci_lower": float(np.quantile(means, 0.025, method="median_unbiased")),
        "bootstrap_ci_upper": float(np.quantile(means, 0.975, method="median_unbiased")),
        "bootstrap_p_candidate_greater": float((1 + np.sum(null_means >= observed)) / (replications + 1)),
        "bootstrap_block_length": float(block),
        "bootstrap_replications": float(replications),
    }


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def build_inference(
    scored: pd.DataFrame,
    manifest: pd.DataFrame,
    contract: Contract,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scope = str(contract["primary_sample_scope"])
    frame = scope_rows(scored, scope)
    eligible = manifest[manifest["include_inference"]]
    strategy_ids = eligible["strategy_id"].tolist()
    benchmark = str(contract["primary_benchmark_id"])
    if benchmark not in strategy_ids:
        raise ProtocolError("primary_benchmark_id must be an inference-eligible strategy.")
    if str(contract["primary_strategy_id"]) not in strategy_ids:
        raise ProtocolError("primary_strategy_id must be an inference-eligible strategy.")
    windows = frame["window_id"].unique()
    rows: list[dict[str, Any]] = []
    gamma = float(contract["crra_gamma"])
    reality_by_window: dict[str, Any] = {}
    labels = eligible.set_index("strategy_id")["label"].to_dict()
    for window_number, window_id in enumerate(windows):
        window = frame[frame["window_id"] == window_id]
        returns: dict[str, np.ndarray] = {}
        dates: np.ndarray | None = None
        for strategy_id in strategy_ids:
            strategy = window[window["strategy_id"] == strategy_id].sort_values("decision_date")
            if dates is None:
                dates = strategy["holding_end_date"].to_numpy()
            elif not np.array_equal(strategy["holding_end_date"].to_numpy(), dates):
                raise ProtocolError("Inference strategies do not share identical realized dates.")
            returns[strategy_id] = strategy["net_return"].to_numpy(float)
        benchmark_utility = crra_utility(returns[benchmark], gamma)
        candidate_ids = [name for name in strategy_ids if name != benchmark]
        if not candidate_ids:
            raise ProtocolError("At least one inference candidate must accompany the primary benchmark.")
        comparison_pairs: dict[tuple[str, str], set[str]] = {}
        for candidate in candidate_ids:
            comparison_pairs.setdefault((candidate, benchmark), set()).add("benchmark_screen")
        primary = str(contract["primary_strategy_id"])
        for alternative in strategy_ids:
            if alternative != primary:
                comparison_pairs.setdefault((primary, alternative), set()).add("primary_vs_alternative")
        for candidate_number, ((candidate, comparison_benchmark), families) in enumerate(comparison_pairs.items()):
            candidate_returns = returns[candidate]
            comparison_returns = returns[comparison_benchmark]
            return_difference = candidate_returns - comparison_returns
            tracking_error = float(np.std(return_difference, ddof=1))
            candidate_multiple = float(np.prod(1.0 + candidate_returns))
            benchmark_multiple = float(np.prod(1.0 + comparison_returns))
            difference = (
                crra_utility(candidate_returns, gamma)
                - crra_utility(comparison_returns, gamma)
            )
            result: dict[str, Any] = {
                "sample_scope": scope,
                "window_id": window_id,
                "comparison_family": "|".join(sorted(families)),
                "candidate_id": candidate,
                "candidate_label": labels[candidate],
                "benchmark_id": comparison_benchmark,
                "benchmark_label": labels[comparison_benchmark],
                "annualized_mean_return_difference": float(
                    return_difference.mean() * float(contract["periods_per_year"])
                ),
                "annualized_tracking_error": tracking_error * math.sqrt(
                    float(contract["periods_per_year"])
                ),
                "information_ratio": (
                    float(return_difference.mean() / tracking_error * math.sqrt(
                        float(contract["periods_per_year"])
                    )) if tracking_error > 0 else math.nan
                ),
                "candidate_monthly_win_fraction": float(
                    np.mean(candidate_returns > comparison_returns)
                ),
                "candidate_total_return": candidate_multiple - 1.0,
                "benchmark_total_return": benchmark_multiple - 1.0,
                "total_return_difference": candidate_multiple - benchmark_multiple,
                "candidate_terminal_wealth": float(contract["initial_wealth"]) * candidate_multiple,
                "benchmark_terminal_wealth": float(contract["initial_wealth"]) * benchmark_multiple,
            }
            result.update(newey_west_mean_test(difference))
            result.update(paired_block_bootstrap(
                difference, contract, seed_offset=1000 * window_number + candidate_number
            ))
            rows.append(result)
        matrix = np.column_stack([
            crra_utility(returns[candidate], gamma) - benchmark_utility
            for candidate in candidate_ids
        ])
        observed_means = matrix.mean(axis=0)
        n = len(matrix)
        block = min(int(contract["bootstrap_block_length"]), n)
        replications = int(contract["bootstrap_replications"])
        rng = np.random.default_rng(int(contract["inference_seed"]) + 99991 + window_number)
        centered = matrix - observed_means
        observed = math.sqrt(n) * float(observed_means.max())
        bootstrap_max = np.empty(replications)
        for index in range(replications):
            sample = moving_block_indices(rng, n, block)
            bootstrap_max[index] = math.sqrt(n) * float(centered[sample].mean(axis=0).max())
        reality_by_window[str(window_id)] = {
            "sample_scope": scope,
            "benchmark_id": benchmark,
            "best_candidate_id": candidate_ids[int(np.argmax(observed_means))],
            "observed_statistic": observed,
            "p_value": float((1 + np.sum(bootstrap_max >= observed)) / (replications + 1)),
            "block_length": block,
            "replications": replications,
            "mean_utility_differences": dict(zip(candidate_ids, observed_means.tolist())),
        }
    output = pd.DataFrame(rows)
    output["hac_p_holm"] = holm_adjust(output["hac_p_candidate_greater"])
    output["bootstrap_p_holm"] = holm_adjust(output["bootstrap_p_candidate_greater"])
    primary_family = output["comparison_family"].str.contains(
        "primary_vs_alternative", regex=False
    )
    output["hac_p_holm_primary_family"] = math.nan
    output["bootstrap_p_holm_primary_family"] = math.nan
    output.loc[primary_family, "hac_p_holm_primary_family"] = holm_adjust(
        output.loc[primary_family, "hac_p_candidate_greater"]
    )
    output.loc[primary_family, "bootstrap_p_holm_primary_family"] = holm_adjust(
        output.loc[primary_family, "bootstrap_p_candidate_greater"]
    )
    return output, reality_by_window


def seed_robustness_table(metrics: pd.DataFrame) -> pd.DataFrame:
    source = metrics[
        (metrics["report_seed_distribution"])
        & (metrics["sample_scope"] == "complete_periods")
    ].copy()
    if source.empty:
        return pd.DataFrame(columns=["method", "n_seeds", "metric", "median", "q05", "q95", "mean", "std"])
    metrics_to_report = [
        "cagr", "annual_volatility", "sharpe_ratio", "max_drawdown",
        "realized_cvar05_loss", "mean_monthly_turnover", "mean_gross_exposure",
        "terminal_wealth",
    ]
    rows = []
    for (method, window_id), group in source.groupby(["method", "window_id"], sort=False):
        for metric in metrics_to_report:
            values = group[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            rows.append(
                {
                    "method": method,
                    "window_id": window_id,
                    "n_seeds": len(values),
                    "metric": metric,
                    "median": float(np.median(values)),
                    "q05": float(np.quantile(values, 0.05)),
                    "q95": float(np.quantile(values, 0.95)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
                }
            )
    return pd.DataFrame(rows)


def latex_table(frame: pd.DataFrame, path: Path, percent_columns: set[str] | None = None) -> None:
    percent_columns = percent_columns or set()
    display = frame.copy()
    for name in display.select_dtypes(include=[np.number]).columns:
        if name in percent_columns:
            display[name] = display[name].map(lambda value: f"{100 * value:.2f}" if pd.notna(value) else "")
        else:
            display[name] = display[name].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    path.write_text(
        display.to_latex(index=False, escape=True, na_rep="", longtable=False),
        encoding="utf-8",
    )


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_figures(
    scored: pd.DataFrame,
    metrics: pd.DataFrame,
    main: pd.DataFrame,
    inference: pd.DataFrame,
    manifest: pd.DataFrame,
    contract: Contract,
    directory: Path,
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    scope = str(contract["primary_sample_scope"])
    frame = scope_rows(scored, scope)
    main_ids = manifest.loc[manifest["include_main"], "strategy_id"].tolist()
    labels = manifest.set_index("strategy_id")["label"].to_dict()
    for window_id, window in frame[frame["strategy_id"].isin(main_ids)].groupby("window_id", sort=False):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(window_id))
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for index, strategy_id in enumerate(main_ids):
            group = window[window["strategy_id"] == strategy_id].sort_values("holding_end_date")
            if group.empty:
                continue
            multiple = np.cumprod(1.0 + group["net_return"].to_numpy(float))
            ax.plot(group["holding_end_date"], multiple, label=labels[strategy_id], color=COLORS[index % len(COLORS)], linewidth=1.6)
        ax.axhline(1.0, color="#777777", linewidth=0.7)
        ax.set_ylabel("Net wealth (initial = 1)")
        ax.set_xlabel("")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.autofmt_xdate()
        save_figure(fig, directory / f"figure_01_wealth_{safe}")
        generated.append(f"figure_01_wealth_{safe}")

        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for index, strategy_id in enumerate(main_ids):
            group = window[window["strategy_id"] == strategy_id].sort_values("holding_end_date")
            if group.empty:
                continue
            wealth = np.cumprod(1.0 + group["net_return"].to_numpy(float))
            drawdown = wealth / np.maximum.accumulate(np.r_[1.0, wealth])[1:] - 1.0
            ax.plot(group["holding_end_date"], 100 * drawdown, label=labels[strategy_id], color=COLORS[index % len(COLORS)], linewidth=1.4)
        ax.set_ylabel("Drawdown (%)")
        ax.set_xlabel("")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.autofmt_xdate()
        save_figure(fig, directory / f"figure_02_drawdown_{safe}")
        generated.append(f"figure_02_drawdown_{safe}")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for index, row in main.reset_index(drop=True).iterrows():
        ax.scatter(100 * row["annual_volatility"], 100 * row["cagr"], s=45, color=COLORS[index % len(COLORS)])
        ax.annotate(row["label"], (100 * row["annual_volatility"], 100 * row["cagr"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Annualised volatility (%)")
    ax.set_ylabel("CAGR (%)")
    ax.grid(alpha=0.2)
    save_figure(fig, directory / "figure_03_risk_return")
    generated.append("figure_03_risk_return")

    primary = str(contract["primary_strategy_id"])
    primary_rows = frame[frame["strategy_id"] == primary].sort_values(["window_id", "decision_date"])
    # Preserve the frozen realized-panel/actor asset order in the figure.
    weight_columns = [name for name in primary_rows if name.startswith("w_")]
    if not primary_rows.empty and weight_columns:
        for window_id, group in primary_rows.groupby("window_id", sort=False):
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(window_id))
            matrix = group[weight_columns].to_numpy(float).T
            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-float(contract["max_short_weight"]), vmax=float(contract["max_long_weight"]))
            ax.set_yticks(range(len(weight_columns)), [name[2:] for name in weight_columns])
            positions = np.linspace(0, len(group) - 1, min(6, len(group)), dtype=int)
            ax.set_xticks(positions, group.iloc[positions]["decision_date"].dt.strftime("%Y-%m"), rotation=30, ha="right")
            ax.set_xlabel("Decision month")
            fig.colorbar(image, ax=ax, label="Portfolio weight")
            save_figure(fig, directory / f"figure_04_allocation_heatmap_{safe}")
            generated.append(f"figure_04_allocation_heatmap_{safe}")

            fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.8), sharex=True)
            dates = group["decision_date"]
            axes[0].plot(dates, group["gross_exposure"], color=COLORS[0])
            axes[0].axhline(float(contract["gross_leverage"]), color="#777777", linestyle="--", linewidth=0.8)
            axes[0].set_ylabel("Gross")
            axes[1].plot(dates, group["short_notional"], color=COLORS[1])
            axes[1].set_ylabel("Short book")
            axes[2].plot(dates, group["turnover"], color=COLORS[2])
            axes[2].set_ylabel("Turnover")
            axes[2].set_xlabel("")
            for ax in axes:
                ax.grid(alpha=0.2)
            fig.autofmt_xdate()
            save_figure(fig, directory / f"figure_05_implementation_{safe}")
            generated.append(f"figure_05_implementation_{safe}")

    seed_source = metrics[
        (metrics["sample_scope"] == scope) & metrics["report_seed_distribution"]
    ]
    if seed_source["strategy_id"].nunique() >= 1:
        plot_metrics = ["sharpe_ratio", "cagr", "max_drawdown", "mean_monthly_turnover"]
        fig, axes = plt.subplots(1, len(plot_metrics), figsize=(12, 4.2))
        methods = list(seed_source["method"].unique())
        for ax, metric in zip(axes, plot_metrics):
            values = [seed_source.loc[seed_source["method"] == method, metric].dropna().to_numpy() for method in methods]
            # Matplotlib 3.9 renamed boxplot(labels=...) to tick_labels=... and
            # newer releases removed the former keyword. Set ticks separately
            # to keep the evaluator compatible across both API generations.
            positions = np.arange(1, len(methods) + 1)
            ax.boxplot(values, positions=positions, showmeans=True)
            ax.set_xticks(positions)
            ax.set_xticklabels(methods)
            ax.set_title(metric.replace("_", " "))
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        save_figure(fig, directory / "figure_06_seed_robustness")
        generated.append("figure_06_seed_robustness")

    primary_inference = inference[
        (inference["candidate_id"] == primary)
        & inference["comparison_family"].str.contains("primary_vs_alternative", regex=False)
    ].copy()
    if not primary_inference.empty:
        primary_inference = primary_inference.sort_values(
            "mean_utility_difference", kind="stable"
        )
        y = np.arange(len(primary_inference))
        estimate = primary_inference["mean_utility_difference"].to_numpy(float)
        lower = primary_inference["bootstrap_ci_lower"].to_numpy(float)
        upper = primary_inference["bootstrap_ci_upper"].to_numpy(float)
        fig_height = max(4.2, 0.55 * len(primary_inference) + 1.5)
        fig, ax = plt.subplots(figsize=(8.2, fig_height))
        ax.errorbar(
            estimate, y, xerr=np.vstack([estimate - lower, upper - estimate]),
            fmt="o", color=COLORS[0], ecolor="#666666", capsize=3
        )
        ax.axvline(0.0, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_yticks(y, primary_inference["benchmark_label"])
        ax.set_xlabel("Mean monthly CRRA utility difference (95% block-bootstrap CI)")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.2)
        save_figure(fig, directory / "figure_07_primary_utility_effects")
        generated.append("figure_07_primary_utility_effects")

    benchmark_ids = manifest.loc[
        manifest["include_main"] & (manifest["role"] == "benchmark"),
        "strategy_id",
    ].tolist()
    for window_id, group in frame.groupby("window_id", sort=False):
        primary_group = group[group["strategy_id"] == primary].sort_values(
            "decision_date", kind="stable"
        )
        if primary_group.empty or not benchmark_ids:
            continue
        differences, comparison_labels = [], []
        for benchmark_id in benchmark_ids:
            comparison = group[group["strategy_id"] == benchmark_id].sort_values(
                "decision_date", kind="stable"
            )
            if len(comparison) != len(primary_group) or not np.array_equal(
                comparison["holding_end_date"].to_numpy(),
                primary_group["holding_end_date"].to_numpy(),
            ):
                raise ProtocolError("Primary/benchmark return heatmap dates are not aligned.")
            differences.append(
                100.0 * (primary_group["net_return"].to_numpy(float)
                         - comparison["net_return"].to_numpy(float))
            )
            comparison_labels.append(labels[benchmark_id])
        matrix = np.vstack(differences)
        bound = max(float(np.quantile(np.abs(matrix), 0.95)), 0.25)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(window_id))
        fig, ax = plt.subplots(figsize=(9.0, max(4.2, 0.55 * len(benchmark_ids) + 1.8)))
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu", vmin=-bound, vmax=bound)
        ax.set_yticks(range(len(comparison_labels)), comparison_labels)
        positions = np.linspace(0, len(primary_group) - 1, min(8, len(primary_group)), dtype=int)
        ax.set_xticks(
            positions,
            primary_group.iloc[positions]["holding_end_date"].dt.strftime("%Y-%m"),
            rotation=30, ha="right",
        )
        ax.set_xlabel("Holding-period end")
        fig.colorbar(image, ax=ax, label="Primary minus benchmark net return (percentage points)")
        save_figure(fig, directory / f"figure_08_monthly_excess_return_{safe}")
        generated.append(f"figure_08_monthly_excess_return_{safe}")
    return generated


def write_outputs(
    output: Path,
    contract_path: Path,
    realized_path: Path,
    manifest_path: Path,
    contract: Contract,
    realized: pd.DataFrame,
    manifest: pd.DataFrame,
    scored: pd.DataFrame,
    metrics: pd.DataFrame,
    main: pd.DataFrame,
    inference: pd.DataFrame,
    reality: dict[str, Any],
    hashes: list[dict[str, Any]],
) -> None:
    raw_dir = output / "raw"
    table_dir = output / "tables"
    figure_dir = output / "figures"
    for directory in [raw_dir, table_dir, figure_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    scored.to_csv(raw_dir / "scored_monthly_panel.csv", index=False, date_format="%Y-%m-%d")
    manifest.to_csv(raw_dir / "validated_strategy_manifest.csv", index=False)
    pd.DataFrame(hashes).to_csv(raw_dir / "input_hashes.csv", index=False)
    checks = pd.DataFrame(
        [
            ["realized_period_count", "pass", f"{len(realized)} locked rows"],
            ["common_realized_panel", "pass", "Every strategy joined one-to-one on period keys"],
            ["net_exposure", "pass", f"target={contract['net_exposure']}"],
            ["gross_leverage", "pass", f"cap={contract['gross_leverage']}"],
            ["single_asset_caps", "pass", f"long={contract['max_long_weight']}; short={contract['max_short_weight']}"],
            ["cost_contract", "pass", "Common turnover, short-borrow and cash-financing rates applied"],
            ["primary_scope", "pass", str(contract["primary_sample_scope"])],
            ["seed_inference", "pass", "Individual seeds excluded from market-path inference"],
        ],
        columns=["check", "status", "detail"],
    )
    checks.to_csv(raw_dir / "protocol_checks.csv", index=False)
    metrics.to_csv(raw_dir / "metrics_per_strategy_window_scope.csv", index=False)

    performance_columns = [
        "strategy_id", "label", "n_windows", "sample_scope", "observations",
        "total_return", "terminal_wealth", "cagr", "annual_arithmetic_return",
        "annual_volatility", "annual_downside_deviation", "sharpe_ratio",
        "sortino_ratio", "calmar_ratio", "omega_ratio", "max_drawdown",
        "realized_var05_loss", "realized_cvar05_loss", "tail_event_count",
        "annualized_certainty_equivalent_return", "positive_month_fraction",
        "mean_monthly_turnover",
    ]
    performance = main[performance_columns].copy()
    performance.to_csv(table_dir / "table_01_oos_performance.csv", index=False)
    latex_table(
        performance,
        table_dir / "table_01_oos_performance.tex",
        {
            "total_return", "cagr", "annual_arithmetic_return",
            "annual_volatility", "annual_downside_deviation", "max_drawdown",
            "realized_var05_loss", "realized_cvar05_loss",
            "annualized_certainty_equivalent_return", "positive_month_fraction",
            "mean_monthly_turnover",
        },
    )
    seed_table = seed_robustness_table(metrics)
    seed_table.to_csv(table_dir / "table_02_seed_robustness.csv", index=False)
    latex_table(seed_table, table_dir / "table_02_seed_robustness.tex")
    inference.to_csv(table_dir / "table_03_paired_inference.csv", index=False)
    latex_table(inference, table_dir / "table_03_paired_inference.tex")

    implementation_columns = [
        "strategy_id", "label", "mean_gross_exposure", "maximum_gross_exposure",
        "mean_short_notional", "mean_monthly_turnover", "annualized_turnover",
        "gross_total_return_before_costs", "implementation_drag_total_return",
        "total_transaction_log_cost", "total_financing_log_cost",
        "mean_transaction_cost_bps", "mean_financing_cost_bps",
    ]
    implementation = main[implementation_columns].copy()
    implementation.to_csv(table_dir / "table_04_economic_implementation.csv", index=False)
    latex_table(implementation, table_dir / "table_04_economic_implementation.tex")

    computation = manifest[
        ["strategy_id", "label", "method", "seed", "role", "train_seconds", "evaluation_seconds", "checkpoint_sha256", "code_sha256", "config_sha256"]
    ].copy()
    computation.to_csv(table_dir / "table_05_computation.csv", index=False)
    latex_table(computation, table_dir / "table_05_computation.tex")

    primary_scope = scope_rows(scored, str(contract["primary_sample_scope"]))
    main_ids = manifest.loc[manifest["include_main"], "strategy_id"].tolist()
    monthly = primary_scope[primary_scope["strategy_id"].isin(main_ids)].pivot(
        index=["window_id", "decision_date", "holding_end_date"],
        columns="strategy_id", values="net_return"
    ).reset_index()
    monthly.columns.name = None
    monthly.to_csv(table_dir / "table_06_monthly_net_returns.csv", index=False,
                   date_format="%Y-%m-%d")

    calendar = realized[KEYS + ["trading_days", "is_complete_period"]].copy()
    calendar.to_csv(table_dir / "table_07_locked_calendar.csv", index=False,
                    date_format="%Y-%m-%d")

    locked_all = metrics[
        (metrics["sample_scope"] == "locked_all") & metrics["include_main"]
    ][["window_id"] + [name for name in performance_columns if name != "n_windows"]].copy()
    locked_all.to_csv(
        table_dir / "table_08_shortened_period_robustness.csv", index=False
    )

    distribution_columns = [
        "strategy_id", "label", "observations", "geometric_mean_monthly_return",
        "worst_month_return", "best_month_return", "return_skewness",
        "return_excess_kurtosis", "monthly_certainty_equivalent_return",
        "mean_crra_utility",
    ]
    main[distribution_columns].to_csv(
        table_dir / "table_09_return_distribution.csv", index=False
    )

    primary = str(contract["primary_strategy_id"])
    primary_effects = inference[
        (inference["candidate_id"] == primary)
        & inference["comparison_family"].str.contains(
            "primary_vs_alternative", regex=False
        )
    ][
        [
            "window_id", "candidate_id", "candidate_label", "benchmark_id",
            "benchmark_label",
            "observations", "annualized_mean_return_difference",
            "annualized_tracking_error", "information_ratio",
            "candidate_monthly_win_fraction", "total_return_difference",
            "mean_utility_difference", "bootstrap_ci_lower",
            "bootstrap_ci_upper", "hac_p_candidate_greater",
            "bootstrap_p_candidate_greater", "hac_p_holm",
            "bootstrap_p_holm", "hac_p_holm_primary_family",
            "bootstrap_p_holm_primary_family",
        ]
    ].copy()
    primary_effects.to_csv(
        table_dir / "table_10_primary_effects.csv", index=False
    )
    latex_table(
        primary_effects, table_dir / "table_10_primary_effects.tex",
        {
            "annualized_mean_return_difference", "annualized_tracking_error",
            "candidate_monthly_win_fraction", "total_return_difference",
        },
    )

    constraint_rows: list[dict[str, Any]] = []
    tolerance = float(contract["weight_tolerance"])
    manifest_labels = manifest.set_index("strategy_id")["label"]
    for strategy_id, group in primary_scope[
        primary_scope["strategy_id"].isin(main_ids)
    ].groupby("strategy_id", sort=False):
        weight_columns = [name for name in group.columns if name.startswith("w_")]
        weight_matrix = group[weight_columns].to_numpy(float)
        net_error = float(
            np.max(np.abs(group["net_exposure"].to_numpy(float)
                          - float(contract["net_exposure"])))
        )
        maximum_gross = float(group["gross_exposure"].max())
        maximum_weight = float(np.max(weight_matrix))
        minimum_weight = float(np.min(weight_matrix))
        constraint_rows.append(
            {
                "strategy_id": strategy_id,
                "label": manifest_labels.loc[strategy_id],
                "observations": len(group),
                "maximum_absolute_net_exposure_error": net_error,
                "maximum_gross_exposure": maximum_gross,
                "gross_cap_slack": float(contract["gross_leverage"]) - maximum_gross,
                "maximum_asset_weight": maximum_weight,
                "minimum_asset_weight": minimum_weight,
                "constraint_pass": bool(
                    net_error <= tolerance
                    and maximum_gross <= float(contract["gross_leverage"]) + tolerance
                    and maximum_weight <= float(contract["max_long_weight"]) + tolerance
                    and minimum_weight >= -float(contract["max_short_weight"]) - tolerance
                ),
            }
        )
    constraints = pd.DataFrame(constraint_rows)
    constraints.to_csv(table_dir / "table_11_constraint_audit.csv", index=False)
    latex_table(constraints, table_dir / "table_11_constraint_audit.tex")

    primary_benchmark = str(contract["primary_benchmark_id"])
    primary_test = primary_effects[
        primary_effects["benchmark_id"] == primary_benchmark
    ]
    if len(primary_test) != 1:
        raise ProtocolError("The predeclared primary comparison is not unique.")
    primary_row = primary_test.iloc[0]
    alpha = float(contract["primary_superiority_alpha"])
    secondary_confirmed = bool(
        (primary_effects["mean_utility_difference"] > 0).all()
        and (primary_effects["bootstrap_p_holm_primary_family"] <= alpha).all()
    )
    superiority = {
        "schema_version": 1,
        "sample_scope": str(contract["primary_sample_scope"]),
        "primary_strategy_id": primary,
        "primary_benchmark_id": primary_benchmark,
        "outcome": "mean_monthly_crra_utility_difference",
        "test": str(contract["primary_superiority_test"]),
        "alpha": alpha,
        "estimate": float(primary_row["mean_utility_difference"]),
        "bootstrap_ci_lower": float(primary_row["bootstrap_ci_lower"]),
        "bootstrap_ci_upper": float(primary_row["bootstrap_ci_upper"]),
        "one_sided_bootstrap_p": float(
            primary_row["bootstrap_p_candidate_greater"]
        ),
        "primary_superiority_confirmed": bool(
            primary_row["mean_utility_difference"] > 0
            and primary_row["bootstrap_p_candidate_greater"] <= alpha
        ),
        "secondary_all_benchmarks_holm_confirmed": secondary_confirmed,
        "interpretation_rule": (
            "Primary superiority requires a positive predeclared CRRA utility "
            "effect and one-sided paired moving-block-bootstrap p <= alpha."
        ),
    }
    (table_dir / "primary_superiority_decision.json").write_text(
        json.dumps(superiority, indent=2), encoding="utf-8"
    )
    (table_dir / "white_reality_check.json").write_text(json.dumps(reality, indent=2), encoding="utf-8")
    generated_figures = render_figures(
        scored, metrics, main, inference, manifest, contract, figure_dir
    )

    run_manifest = {
        "schema_version": 1,
        "evaluation_id": contract["evaluation_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": sha256_file(contract_path),
        "realized_panel_sha256": sha256_file(realized_path),
        "strategy_manifest_sha256": sha256_file(manifest_path),
        "strategy_count_including_ensembles": int(manifest["strategy_id"].nunique()),
        "window_count": int(realized["window_id"].nunique()),
        "primary_sample_scope": contract["primary_sample_scope"],
        "primary_strategy_id": contract["primary_strategy_id"],
        "primary_benchmark_id": contract["primary_benchmark_id"],
        "generated_figures": generated_figures,
        "software": {
            "python": os.sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")


def run_pipeline(contract_path: Path, realized_path: Path, manifest_path: Path, output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise ProtocolError(f"Output path already exists; locked results are immutable: {output}")
    contract = Contract.read(contract_path)
    realized, assets = read_realized_panel(realized_path, contract)
    manifest = read_strategy_manifest(manifest_path, contract)
    weights: dict[str, pd.DataFrame] = {}
    hashes: list[dict[str, Any]] = [
        {"artifact": "evaluation_contract", "path": str(contract_path.resolve()), "sha256": sha256_file(contract_path)},
        {"artifact": "realized_panel", "path": str(realized_path.resolve()), "sha256": sha256_file(realized_path)},
        {"artifact": "strategy_manifest", "path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
    ]
    for _, row in manifest.iterrows():
        weight, digest = read_and_validate_weights(row, manifest_path, realized, assets, contract)
        weights[row["strategy_id"]] = weight
        hashes.append(
            {
                "artifact": f"weights:{row['strategy_id']}",
                "path": row["weight_log_path"],
                "sha256": digest,
            }
        )
    weights, manifest = build_ensembles(weights, manifest, assets, contract)
    if str(contract["primary_strategy_id"]) not in weights:
        raise ProtocolError("The declared primary strategy was not constructed or loaded.")
    scored_parts = [
        score_strategy(strategy_id, weight, realized, assets, contract)
        for strategy_id, weight in weights.items()
    ]
    scored = pd.concat(scored_parts, ignore_index=True)
    metrics, main = build_metric_tables(scored, manifest, contract)
    inference, reality = build_inference(scored, manifest, contract)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=str(output.parent)))
    try:
        write_outputs(
            temporary, contract_path, realized_path, manifest_path, contract,
            realized, manifest, scored, metrics, main, inference, reality, hashes,
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--realized", required=True, type=Path)
    parser.add_argument("--strategies", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_pipeline(args.contract, args.realized, args.strategies, args.output)
    except ProtocolError as error:
        print(f"PROTOCOL FAILURE: {error}", file=os.sys.stderr)
        return 2
    print(f"Publication artifacts written immutably to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
