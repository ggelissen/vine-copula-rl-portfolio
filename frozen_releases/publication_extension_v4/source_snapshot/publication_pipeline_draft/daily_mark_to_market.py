#!/usr/bin/env python3
"""Reconstruct daily mark-to-market risk from frozen monthly target weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from publication_pipeline_draft.publication_pipeline import (
    Contract, ProtocolError, build_ensembles, read_and_validate_weights,
    read_realized_panel, read_strategy_manifest,
)


class DailyRiskError(RuntimeError):
    pass


KEYS = ["window_id", "decision_date", "holding_end_date"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DailyRiskError(message)


def daily_metrics(group: pd.DataFrame) -> dict[str, float]:
    returns = group["net_return"].to_numpy(float)
    require(len(returns) >= 100 and np.isfinite(returns).all() and
            (returns > -1).all(), "Daily tail diagnostics need at least 100 valid days.")
    wealth = np.cumprod(1 + returns)
    drawdown = wealth / np.maximum.accumulate(np.r_[1.0, wealth])[1:] - 1
    losses = -returns
    var = float(np.quantile(losses, 0.95, method="linear"))
    tail = losses[losses >= var]
    negative = np.minimum(returns, 0)
    return {
        "daily_observations": len(returns),
        "annualized_daily_volatility": float(np.std(returns, ddof=1) * np.sqrt(252)),
        "annualized_daily_downside_deviation": float(
            np.sqrt(np.mean(negative ** 2)) * np.sqrt(252)),
        "daily_var05_loss": var,
        "daily_cvar05_loss": float(tail.mean()),
        "daily_tail_event_count": len(tail),
        "worst_daily_return": float(returns.min()),
        "best_daily_return": float(returns.max()),
        "daily_path_max_drawdown": float(-drawdown.min()),
    }


def run(contract_path: Path, realized_path: Path, strategy_path: Path,
        daily_path: Path, return_manifest_path: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"Output already exists: {output}")
    contract = Contract.read(contract_path)
    require(contract.get("turnover_convention") == "drifted_pretrade_v1" and
            contract.get("financing_proration") == "actual_calendar_days_v1",
            "Daily audit requires drifted turnover and actual-day financing.")
    return_manifest = json.loads(return_manifest_path.read_text(encoding="utf-8"))
    require(return_manifest.get("return_file_sha256") == sha256(daily_path),
            "Daily return file hash differs from its frozen manifest.")
    daily = pd.read_csv(daily_path)
    require("date" in daily, "Daily return panel needs a date column.")
    assets = list(return_manifest["asset_order"])
    require([name for name in daily if name != "date"] == assets,
            "Daily return asset order differs from its frozen manifest.")
    daily["date"] = pd.to_datetime(daily["date"], errors="raise").dt.normalize()
    values = daily[assets].apply(pd.to_numeric, errors="raise").to_numpy(float)
    require(np.isfinite(values).all() and not daily["date"].duplicated().any(),
            "Daily return panel is non-finite or duplicated.")
    daily[assets] = values

    realized, realized_assets = read_realized_panel(realized_path, contract)
    require(realized_assets == assets, "Daily and monthly realized assets differ.")
    manifest = read_strategy_manifest(strategy_path, contract)
    weights: dict[str, pd.DataFrame] = {}
    for _, row in manifest.iterrows():
        value, _ = read_and_validate_weights(
            row, strategy_path, realized, assets, contract)
        weights[row["strategy_id"]] = value
    weights, manifest = build_ensembles(weights, manifest, assets, contract)
    selected = manifest.loc[
        manifest["include_main"] | manifest["report_seed_distribution"],
        "strategy_id",
    ].tolist()
    weight_columns = [f"w_{asset}" for asset in assets]
    gross_columns = [f"g_{asset}" for asset in assets]
    daily_rows: list[dict[str, Any]] = []
    reconciliation: list[dict[str, Any]] = []
    for strategy_id in selected:
        merged = realized.merge(weights[strategy_id], on=KEYS,
                                how="inner", validate="one_to_one")
        for window_id, window in merged.groupby("window_id", sort=False):
            pretrade = np.repeat(float(contract["net_exposure"]) / len(assets),
                                 len(assets))
            for _, period in window.sort_values("decision_date").iterrows():
                target = period[weight_columns].to_numpy(float)
                gross_month = period[gross_columns].to_numpy(float)
                turnover = float(np.abs(target - pretrade).sum())
                transaction = float(contract["turnover_cost"]) * turnover
                short_notional = float(np.maximum(-target, 0).sum())
                cash_borrow = max(float(target.sum()) - 1.0, 0.0)
                calendar_days = (period["holding_end_date"] -
                                 period["decision_date"]).days
                financing = (
                    float(contract["annual_short_borrow_rate"]) * short_notional +
                    float(contract["annual_cash_borrow_rate"]) * cash_borrow
                ) * calendar_days / float(contract.get("day_count_basis", 365))
                mask = ((daily["date"] > period["decision_date"]) &
                        (daily["date"] <= period["holding_end_date"]))
                holding = daily.loc[mask]
                require(len(holding) == int(period["trading_days"]),
                        "Daily/monthly trading-day counts differ.")
                current = target.copy()
                net_values = []
                gross_values = []
                for day_number, (_, day) in enumerate(holding.iterrows()):
                    asset_gross = np.exp(day[assets].to_numpy(float))
                    portfolio_gross = 1 + float(np.dot(current, asset_gross - 1))
                    require(math.isfinite(portfolio_gross) and portfolio_gross > 0,
                            "Daily portfolio path became insolvent.")
                    daily_log_cost = financing / len(holding)
                    if day_number == 0:
                        daily_log_cost += transaction
                    net_gross = portfolio_gross * math.exp(-daily_log_cost)
                    gross_values.append(portfolio_gross)
                    net_values.append(net_gross)
                    daily_rows.append({
                        "strategy_id": strategy_id, "window_id": window_id,
                        "date": day["date"], "decision_date": period["decision_date"],
                        "holding_end_date": period["holding_end_date"],
                        "gross_return": portfolio_gross - 1,
                        "net_return": net_gross - 1,
                        "transaction_log_cost": transaction if day_number == 0 else 0,
                        "financing_log_cost": financing / len(holding),
                        "gross_exposure": float(np.abs(current).sum()),
                        "short_notional": float(np.maximum(-current, 0).sum()),
                    })
                    current = current * asset_gross / portfolio_gross
                expected_gross = 1 + float(np.dot(target, gross_month - 1))
                observed_gross = float(np.prod(gross_values))
                expected_net = expected_gross * math.exp(-transaction - financing)
                observed_net = float(np.prod(net_values))
                require(abs(expected_gross - observed_gross) <= 1e-10 and
                        abs(expected_net - observed_net) <= 1e-10,
                        "Daily path does not reconcile to common monthly accounting.")
                reconciliation.append({
                    "strategy_id": strategy_id, "window_id": window_id,
                    "holding_end_date": period["holding_end_date"],
                    "gross_error": observed_gross - expected_gross,
                    "net_error": observed_net - expected_net,
                })
                pretrade = target * gross_month / expected_gross
    daily_frame = pd.DataFrame(daily_rows)
    metadata = manifest.set_index("strategy_id")
    metric_rows = []
    for (strategy_id, window_id), group in daily_frame.groupby(
            ["strategy_id", "window_id"], sort=False):
        metric_rows.append({
            "strategy_id": strategy_id, "label": metadata.loc[strategy_id, "label"],
            "method": metadata.loc[strategy_id, "method"],
            "window_id": window_id, **daily_metrics(group),
        })
    metric_frame = pd.DataFrame(metric_rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        daily_frame.to_csv(temporary / "daily_strategy_returns.csv", index=False,
                           date_format="%Y-%m-%d")
        metric_frame.to_csv(temporary / "daily_tail_risk_metrics.csv", index=False)
        pd.DataFrame(reconciliation).to_csv(
            temporary / "monthly_reconciliation.csv", index=False)
        manifest_value = {
            "schema_version": 1, "status": "daily_mark_to_market_complete",
            "strategy_count": len(selected),
            "all_monthly_paths_reconciled": True,
            "cost_allocation": (
                "rebalance transaction cost on first trading day; monthly target-based "
                "financing log cost allocated uniformly over holding-period trading days"),
            "daily_returns_sha256": sha256(daily_path),
            "confirmatory_claim_permitted": bool(
                contract.get("confirmatory_claim_permitted", False)),
        }
        (temporary / "daily_risk_manifest.json").write_text(
            json.dumps(manifest_value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        os.replace(temporary, output)
        return manifest_value
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--realized", required=True, type=Path)
    parser.add_argument("--strategies", required=True, type=Path)
    parser.add_argument("--daily-returns", required=True, type=Path)
    parser.add_argument("--return-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args.contract.resolve(), args.realized.resolve(),
                     args.strategies.resolve(), args.daily_returns.resolve(),
                     args.return_manifest.resolve(), args.output)
    except (OSError, ValueError, json.JSONDecodeError, ProtocolError,
            DailyRiskError) as error:
        print(f"DAILY RISK FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
