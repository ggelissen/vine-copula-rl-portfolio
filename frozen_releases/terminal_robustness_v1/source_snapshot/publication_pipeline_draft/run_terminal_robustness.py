#!/usr/bin/env python3
"""Execute deterministic daily-risk, friction, and inference robustness analyses."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from publication_pipeline_draft.terminal_robustness_protocol import (
    FROZEN_CODE, TerminalProtocolError, require, sha256, verify_release,
    write_contents,
)


class TerminalAnalysisError(RuntimeError):
    pass


KEYS = ["window_id", "decision_date", "holding_end_date"]


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values = series.astype(str).str.strip().str.lower()
    require(values.isin({"true", "false", "1", "0", "yes", "no"}).all(),
            "is_complete_period contains an invalid Boolean marker.")
    return values.isin({"true", "1", "yes"})


def crra_utility(returns: np.ndarray, gamma: float) -> np.ndarray:
    gross = 1.0 + np.asarray(returns, dtype=float)
    require(np.isfinite(gross).all() and bool((gross > 0).all()),
            "CRRA utility requires positive finite gross returns.")
    if gamma == 1.0:
        return np.log(gross)
    return (gross ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def annualized_ce(returns: np.ndarray, gamma: float, factor: float) -> float:
    mean_utility = float(crra_utility(returns, gamma).mean())
    if gamma == 1.0:
        gross = math.exp(mean_utility)
    else:
        base = 1.0 + (1.0 - gamma) * mean_utility
        require(base > 0, "CRRA certainty-equivalent inversion is undefined.")
        gross = base ** (1.0 / (1.0 - gamma))
    return float(gross ** factor - 1.0)


def annualized_ce_rows(returns: np.ndarray, gamma: float,
                       factor: float) -> np.ndarray:
    gross = 1.0 + returns
    if gamma == 1.0:
        mean_utility = np.log(gross).mean(axis=1)
        ce_gross = np.exp(mean_utility)
    else:
        utility = (gross ** (1.0 - gamma) - 1.0) / (1.0 - gamma)
        base = 1.0 + (1.0 - gamma) * utility.mean(axis=1)
        require(bool((base > 0).all()), "Bootstrap CE inversion is undefined.")
        ce_gross = base ** (1.0 / (1.0 - gamma))
    return ce_gross ** factor - 1.0


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def load_panels(release: Path, contract: dict[str, Any]
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(release / "source_inventory.csv")
    frames: list[pd.DataFrame] = []
    ledger: list[dict[str, Any]] = []
    assets = list(contract["asset_order"])
    weights = [f"w_{asset}" for asset in assets]
    numeric = [
        "gross_return", "net_return", "turnover", "transaction_cost",
        "financing_cost", "short_notional", "cash_borrow_notional",
        "gross_exposure", "net_exposure", *weights,
    ]
    tolerance = float(contract["daily_risk"]["weight_tolerance"])
    for source in contract["sources"]:
        row = inventory[inventory["source_id"] == source["source_id"]]
        require(len(row) == 1, f"Frozen source inventory mismatch: {source['source_id']}")
        path = release / row.iloc[0]["snapshot_path"]
        require(sha256(path) == row.iloc[0]["sha256"],
                f"Frozen evidence panel hash mismatch: {source['source_id']}")
        frame = pd.read_csv(path)
        frame["decision_date"] = pd.to_datetime(
            frame["decision_date"], errors="raise").dt.normalize()
        frame["holding_end_date"] = pd.to_datetime(
            frame["holding_end_date"], errors="raise").dt.normalize()
        frame["is_complete_period"] = parse_bool(frame["is_complete_period"])
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
        require(np.isfinite(frame[numeric].to_numpy(float)).all(),
                f"Non-finite scored panel values: {source['source_id']}")
        require(not frame.duplicated(["strategy_id", *KEYS]).any(),
                f"Duplicate strategy periods: {source['source_id']}")
        matrix = frame[weights].to_numpy(float)
        require(np.max(np.abs(matrix.sum(axis=1) - 1.0)) <= tolerance,
                f"Net weights fail in {source['source_id']}")
        require(np.max(np.abs(matrix).sum(axis=1)) <=
                float(contract["daily_risk"]["gross_exposure_cap"]) + tolerance,
                f"Gross weights fail in {source['source_id']}")
        require(matrix.max() <= float(contract["daily_risk"]["position_long_cap"]) +
                tolerance and matrix.min() >= float(
                    contract["daily_risk"]["position_short_cap"]) - tolerance,
                f"Position weights fail in {source['source_id']}")
        frame.insert(0, "source_id", source["source_id"])
        frame.insert(1, "evidence_class", source["evidence_class"])
        frame.insert(2, "claim_scope", source["claim_scope"])
        frame.insert(3, "canonical_strategy_id", source["source_id"] + "::" +
                     frame["strategy_id"].astype(str))
        frames.append(frame)
        ledger.append({
            "source_id": source["source_id"],
            "evidence_class": source["evidence_class"],
            "claim_scope": source["claim_scope"],
            "strategies": frame["strategy_id"].nunique(),
            "windows": frame["window_id"].nunique(),
            "period_rows": len(frame),
            "complete_period_rows": int(frame["is_complete_period"].sum()),
            "first_decision_date": frame["decision_date"].min().date().isoformat(),
            "last_holding_end_date": frame["holding_end_date"].max().date().isoformat(),
            "input_sha256": sha256(path),
            "confirmatory_claim_created_by_campaign": False,
        })
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, pd.DataFrame(ledger)


def load_daily_gross(release: Path, assets: list[str]) -> pd.DataFrame:
    path = release / "input_snapshots" / "seven_asset_adjusted_levels.csv"
    levels = pd.read_csv(path, encoding="utf-8-sig")
    require(list(levels) == ["date", *assets],
            "Adjusted-level columns differ from the terminal contract.")
    levels["date"] = pd.to_datetime(levels["date"], errors="raise").dt.normalize()
    require(levels["date"].is_monotonic_increasing and
            not levels["date"].duplicated().any(), "Adjusted-level dates are invalid.")
    values = levels[assets].apply(pd.to_numeric, errors="raise")
    require(np.isfinite(values.to_numpy(float)).all() and
            bool((values > 0).all().all()), "Adjusted levels are invalid.")
    gross = values / values.shift(1)
    gross.insert(0, "date", levels["date"])
    return gross.iloc[1:].reset_index(drop=True)


def daily_replay(panel: pd.DataFrame, daily: pd.DataFrame,
                 contract: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = list(contract["asset_order"])
    weight_columns = [f"w_{asset}" for asset in assets]
    tolerance = float(contract["daily_risk"]["monthly_reconciliation_tolerance"])
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for _, period in panel.sort_values(
            ["source_id", "strategy_id", "window_id", "decision_date"]).iterrows():
        mask = ((daily["date"] > period["decision_date"]) &
                (daily["date"] <= period["holding_end_date"]))
        holding = daily.loc[mask]
        require(bool(len(holding)), "A scored period has no daily observations.")
        if "trading_days" in panel and pd.notna(period.get("trading_days")):
            require(len(holding) == int(period["trading_days"]),
                    "Daily observations differ from recorded trading_days.")
        current = period[weight_columns].to_numpy(float)
        gross_values: list[float] = []
        net_values: list[float] = []
        for day_number, (_, day) in enumerate(holding.iterrows()):
            asset_gross = day[assets].to_numpy(float)
            portfolio_gross = 1.0 + float(np.dot(current, asset_gross - 1.0))
            require(math.isfinite(portfolio_gross) and portfolio_gross > 0,
                    "Daily portfolio path became insolvent.")
            log_cost = float(period["financing_cost"]) / len(holding)
            if day_number == 0:
                log_cost += float(period["transaction_cost"])
            net_gross = portfolio_gross * math.exp(-log_cost)
            gross_values.append(portfolio_gross); net_values.append(net_gross)
            rows.append({
                "source_id": period["source_id"],
                "evidence_class": period["evidence_class"],
                "canonical_strategy_id": period["canonical_strategy_id"],
                "strategy_id": period["strategy_id"],
                "window_id": period["window_id"], "date": day["date"],
                "decision_date": period["decision_date"],
                "holding_end_date": period["holding_end_date"],
                "is_complete_period": bool(period["is_complete_period"]),
                "gross_return": portfolio_gross - 1.0,
                "net_return": net_gross - 1.0,
                "transaction_log_cost": (float(period["transaction_cost"])
                                         if day_number == 0 else 0.0),
                "financing_log_cost": float(period["financing_cost"]) / len(holding),
                "gross_exposure": float(np.abs(current).sum()),
                "short_notional": float(np.maximum(-current, 0).sum()),
            })
            current = current * asset_gross / portfolio_gross
        observed_gross = float(np.prod(gross_values))
        observed_net = float(np.prod(net_values))
        expected_gross = 1.0 + float(period["gross_return"])
        expected_net = 1.0 + float(period["net_return"])
        gross_error = observed_gross - expected_gross
        net_error = observed_net - expected_net
        require(abs(gross_error) <= tolerance and abs(net_error) <= tolerance,
                "Daily path does not reconcile to frozen monthly accounting: "
                f"{period['canonical_strategy_id']} {period['holding_end_date'].date()} "
                f"gross_error={gross_error:.3g}, net_error={net_error:.3g}")
        checks.append({
            "source_id": period["source_id"],
            "canonical_strategy_id": period["canonical_strategy_id"],
            "window_id": period["window_id"],
            "holding_end_date": period["holding_end_date"],
            "daily_observations": len(holding), "gross_error": gross_error,
            "net_error": net_error, "reconciliation_pass": True,
        })
    return pd.DataFrame(rows), pd.DataFrame(checks)


def drawdown_duration(wealth: np.ndarray) -> tuple[int, int]:
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    underwater = wealth < peaks - 1e-14
    longest = current = 0
    for value in underwater:
        current = current + 1 if value else 0
        longest = max(longest, current)
    active = 0
    for value in underwater[::-1]:
        if not value:
            break
        active += 1
    return longest, active


def daily_metrics(daily: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    annual_days = float(contract["daily_risk"]["annualization_days"])
    minimum = int(contract["daily_risk"]["minimum_daily_observations"])
    for scope, frame in (("complete_periods", daily[daily["is_complete_period"]]),
                         ("all_available_periods", daily)):
        for keys, group in frame.groupby(
                ["source_id", "evidence_class", "canonical_strategy_id",
                 "strategy_id", "window_id"], sort=True):
            group = group.sort_values("date")
            returns = group["net_return"].to_numpy(float)
            if len(returns) < minimum:
                continue
            wealth = np.cumprod(1.0 + returns)
            peaks = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
            drawdown = wealth / peaks - 1.0
            negative = np.minimum(returns, 0.0)
            longest, active = drawdown_duration(wealth)
            item: dict[str, Any] = {
                "scope": scope, "source_id": keys[0], "evidence_class": keys[1],
                "canonical_strategy_id": keys[2], "strategy_id": keys[3],
                "window_id": keys[4], "daily_observations": len(returns),
                "total_net_return": float(wealth[-1] - 1.0),
                "annualized_daily_volatility": float(
                    np.std(returns, ddof=1) * math.sqrt(annual_days)),
                "annualized_daily_downside_deviation": float(
                    math.sqrt(float(np.mean(negative ** 2))) * math.sqrt(annual_days)),
                "daily_path_max_drawdown": float(-drawdown.min()),
                "maximum_drawdown_duration_trading_days": longest,
                "unrecovered_drawdown_days_at_end": active,
                "worst_daily_return": float(returns.min()),
                "best_daily_return": float(returns.max()),
                "daily_return_skewness": float(pd.Series(returns).skew()),
                "daily_return_excess_kurtosis": float(pd.Series(returns).kurt()),
            }
            losses = -returns
            for probability in contract["daily_risk"]["tail_probabilities"]:
                label = int(round(float(probability) * 100))
                var = float(np.quantile(losses, float(probability), method="linear"))
                tail = losses[losses >= var]
                item[f"daily_var_{label}_loss"] = var
                item[f"daily_cvar_{label}_loss"] = float(tail.mean())
                item[f"daily_tail_{label}_event_count"] = len(tail)
            rows.append(item)
    return pd.DataFrame(rows)


def elapsed_factor(group: pd.DataFrame) -> tuple[float, float]:
    elapsed = float(((group["holding_end_date"] - group["decision_date"]).dt.days /
                     365.0).sum())
    require(elapsed > 0, "Non-positive evaluation duration.")
    return len(group) / elapsed, elapsed


def path_metrics(group: pd.DataFrame, returns: np.ndarray,
                 gamma: float) -> dict[str, float]:
    factor, elapsed = elapsed_factor(group)
    wealth = np.cumprod(1.0 + returns)
    drawdown = np.r_[1.0, wealth] / np.maximum.accumulate(np.r_[1.0, wealth]) - 1.0
    volatility = (float(np.std(returns, ddof=1) * math.sqrt(factor))
                  if len(returns) > 1 else 0.0)
    return {
        "observations": len(returns), "elapsed_years": elapsed,
        "total_return": float(wealth[-1] - 1.0),
        "cagr": float(wealth[-1] ** (1.0 / elapsed) - 1.0),
        "annualized_volatility": volatility,
        "max_drawdown": float(-drawdown.min()),
        "annualized_certainty_equivalent_return": annualized_ce(
            returns, gamma, factor),
    }


def friction_surface(panel: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    economics = contract["economics"]
    gamma = float(economics["crra_gamma"])
    rows: list[dict[str, Any]] = []
    scopes = (("complete_periods", panel[panel["is_complete_period"]]),
              ("all_available_periods", panel))
    for scope, scoped in scopes:
        for keys, group in scoped.groupby(
                ["source_id", "evidence_class", "canonical_strategy_id",
                 "strategy_id", "window_id"], sort=True):
            group = group.sort_values("decision_date")
            year_fraction = ((group["holding_end_date"] - group["decision_date"]).dt.days /
                             float(economics["day_count_basis"])).to_numpy(float)
            gross = 1.0 + group["gross_return"].to_numpy(float)
            for transaction_bps in economics["transaction_cost_bps_grid"]:
                for short_percent in economics["annual_short_borrow_percent_grid"]:
                    for cash_percent in economics["annual_cash_borrow_percent_grid"]:
                        transaction = (float(transaction_bps) / 10000.0 *
                                       group["turnover"].to_numpy(float))
                        financing = year_fraction * (
                            float(short_percent) / 100.0 *
                            group["short_notional"].to_numpy(float) +
                            float(cash_percent) / 100.0 *
                            group["cash_borrow_notional"].to_numpy(float))
                        returns = gross * np.exp(-transaction - financing) - 1.0
                        rows.append({
                            "scope": scope, "source_id": keys[0],
                            "evidence_class": keys[1],
                            "canonical_strategy_id": keys[2], "strategy_id": keys[3],
                            "window_id": keys[4],
                            "transaction_cost_bps_one_way": transaction_bps,
                            "annual_short_borrow_percent": short_percent,
                            "annual_cash_borrow_percent": cash_percent,
                            **path_metrics(group, returns, gamma),
                        })
    return pd.DataFrame(rows)


def find_path(panel: pd.DataFrame, source: str, strategy: str) -> pd.DataFrame:
    result = panel[(panel["source_id"] == source) &
                   (panel["strategy_id"] == strategy) &
                   panel["is_complete_period"]].copy()
    require(bool(len(result)), f"Registered strategy is absent: {source}::{strategy}")
    return result.sort_values(KEYS)


def align_contrast(panel: pd.DataFrame, item: dict[str, Any]
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = find_path(panel, item["candidate_source"], item["candidate"])
    comparator = find_path(panel, item["comparator_source"], item["comparator"])
    left = candidate[KEYS].reset_index(drop=True)
    right = comparator[KEYS].reset_index(drop=True)
    require(left.equals(right), f"Contrast calendar mismatch: {item['contrast_id']}")
    return candidate, comparator


def moving_block_indices(rng: np.random.Generator, replications: int,
                         n: int, block: int) -> np.ndarray:
    blocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=(replications, blocks))
    offsets = np.arange(block)
    return ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(
        replications, -1)[:, :n]


def stationary_indices(rng: np.random.Generator, replications: int,
                       n: int, expected: int) -> np.ndarray:
    result = np.empty((replications, n), dtype=np.int32)
    result[:, 0] = rng.integers(0, n, size=replications)
    probability = 1.0 / float(expected)
    for column in range(1, n):
        restart = rng.random(replications) < probability
        fresh = rng.integers(0, n, size=replications)
        result[:, column] = np.where(
            restart, fresh, (result[:, column - 1] + 1) % n)
    return result


def bootstrap_task(task: dict[str, Any]) -> dict[str, Any]:
    candidate = np.asarray(task["candidate"], dtype=float)
    comparator = np.asarray(task["comparator"], dtype=float)
    n = len(candidate)
    rng = np.random.default_rng(int(task["seed"]))
    if task["method"] == "moving_block":
        indices = moving_block_indices(
            rng, int(task["replications"]), n, int(task["block_length"]))
    else:
        indices = stationary_indices(
            rng, int(task["replications"]), n, int(task["block_length"]))
    gamma = float(task["gamma"]); factor = float(task["factor"])
    utility_difference = crra_utility(candidate, gamma) - crra_utility(
        comparator, gamma)
    observed_utility = float(utility_difference.mean())
    centered = utility_difference - observed_utility
    null = centered[indices].mean(axis=1)
    effects = (annualized_ce_rows(candidate[indices], gamma, factor) -
               annualized_ce_rows(comparator[indices], gamma, factor))
    observed_ce = (annualized_ce(candidate, gamma, factor) -
                   annualized_ce(comparator, gamma, factor))
    replications = int(task["replications"])
    return {
        "contrast_id": task["contrast_id"], "family": task["family"],
        "method": task["method"], "block_length": int(task["block_length"]),
        "observations": n, "annualized_ce_difference": observed_ce,
        "mean_period_crra_utility_difference": observed_utility,
        "ci_lower": float(np.quantile(effects, 0.025, method="median_unbiased")),
        "ci_upper": float(np.quantile(effects, 0.975, method="median_unbiased")),
        "one_sided_p_candidate_greater": float(
            (1 + np.sum(null >= observed_utility)) / (replications + 1)),
        "two_sided_p": float((1 + np.sum(np.abs(null) >= abs(observed_utility))) /
                             (replications + 1)),
        "bootstrap_replications": replications,
    }


def resampling_robustness(panel: pd.DataFrame, contract: dict[str, Any],
                          workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = contract["inference"]
    gamma = float(contract["economics"]["crra_gamma"])
    tasks: list[dict[str, Any]] = []
    for contrast_number, item in enumerate(contract["contrasts"]):
        candidate, comparator = align_contrast(panel, item)
        factor, _ = elapsed_factor(candidate)
        common = {
            "contrast_id": item["contrast_id"], "family": item["family"],
            "candidate": candidate["net_return"].to_numpy(float),
            "comparator": comparator["net_return"].to_numpy(float),
            "gamma": gamma, "factor": factor,
            "replications": int(settings["bootstrap_replications"]),
        }
        for method_number, (method, lengths) in enumerate((
                ("moving_block", settings["moving_block_lengths"]),
                ("stationary", settings["stationary_expected_block_lengths"]))):
            for length in lengths:
                tasks.append({
                    **common, "method": method, "block_length": int(length),
                    "seed": (int(settings["bootstrap_seed"]) +
                             contrast_number * 100003 + method_number * 1009 +
                             int(length) * 37),
                })
    if workers == 1:
        rows = [bootstrap_task(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(bootstrap_task, tasks))
    result = pd.DataFrame(rows).sort_values(
        ["method", "block_length", "family", "contrast_id"]).reset_index(drop=True)
    result["holm_p_candidate_greater"] = np.nan
    for _, index in result.groupby(["method", "block_length", "family"]).groups.items():
        values = result.loc[index, "one_sided_p_candidate_greater"].tolist()
        result.loc[index, "holm_p_candidate_greater"] = holm_adjust(values)
    result["positive_effect_holm_0_05"] = (
        (result["annualized_ce_difference"] > 0) &
        (result["holm_p_candidate_greater"] <= 0.05))
    result["confirmatory_claim_created"] = False

    influence_rows: list[dict[str, Any]] = []
    for item in contract["contrasts"]:
        candidate, comparator = align_contrast(panel, item)
        c = candidate["net_return"].to_numpy(float)
        b = comparator["net_return"].to_numpy(float)
        for omitted in range(len(c)):
            mask = np.arange(len(c)) != omitted
            leave_factor, _ = elapsed_factor(candidate.iloc[np.flatnonzero(mask)])
            influence_rows.append({
                "contrast_id": item["contrast_id"], "family": item["family"],
                "omitted_window_id": candidate.iloc[omitted]["window_id"],
                "omitted_holding_end_date": candidate.iloc[omitted][
                    "holding_end_date"],
                "annualized_ce_difference": (
                    annualized_ce(c[mask], gamma, leave_factor) -
                    annualized_ce(b[mask], gamma, leave_factor)),
                "remaining_periods": int(mask.sum()),
            })
    return result, pd.DataFrame(influence_rows)


def reality_checks(panel: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    settings = contract["inference"]
    gamma = float(contract["economics"]["crra_gamma"])
    rows: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = {}
    for item in contract["contrasts"]:
        by_family.setdefault(item["family"], []).append(item)
    for family, items in sorted(by_family.items()):
        candidate_keys = {(item["candidate_source"], item["candidate"]) for item in items}
        if len(items) < 2 or len(candidate_keys) != 1:
            continue
        differences = []
        labels = []
        for item in items:
            candidate, comparator = align_contrast(panel, item)
            differences.append(crra_utility(
                candidate["net_return"].to_numpy(float), gamma) - crra_utility(
                    comparator["net_return"].to_numpy(float), gamma))
            labels.append(item["contrast_id"])
        matrix = np.column_stack(differences)
        observed_means = matrix.mean(axis=0)
        observed_max = float(observed_means.max())
        centered = matrix - observed_means
        for method, block in (("moving_block", 3), ("stationary", 3)):
            rng = np.random.default_rng(int(settings["bootstrap_seed"]) +
                                        sum(ord(value) for value in family) +
                                        (0 if method == "moving_block" else 700001))
            replications = int(settings["bootstrap_replications"])
            indices = (moving_block_indices(rng, replications, len(matrix), block)
                       if method == "moving_block" else
                       stationary_indices(rng, replications, len(matrix), block))
            maxima = centered[indices, :].mean(axis=1).max(axis=1)
            rows.append({
                "family": family, "method": method, "block_length": block,
                "candidate_source": items[0]["candidate_source"],
                "candidate": items[0]["candidate"],
                "comparators": len(items),
                "best_observed_contrast_id": labels[int(np.argmax(observed_means))],
                "best_observed_mean_utility_difference": observed_max,
                "white_reality_check_p": float(
                    (1 + np.sum(maxima >= observed_max)) / (replications + 1)),
                "bootstrap_replications": replications,
                "confirmatory_claim_created": False,
            })
    return pd.DataFrame(rows)


def break_even_costs(panel: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    economics = contract["economics"]
    gamma = float(economics["crra_gamma"])
    short_rate = float(economics["primary_annual_short_borrow_percent"]) / 100.0
    cash_rate = float(economics["primary_annual_cash_borrow_percent"]) / 100.0
    maximum = float(economics["break_even_transaction_cost_max_bps"])

    def ce_at(group: pd.DataFrame, bps: float) -> float:
        group = group.sort_values(KEYS)
        year_fraction = ((group["holding_end_date"] - group["decision_date"]).dt.days /
                         float(economics["day_count_basis"])).to_numpy(float)
        costs = (bps / 10000.0 * group["turnover"].to_numpy(float) +
                 year_fraction * (short_rate * group["short_notional"].to_numpy(float) +
                                  cash_rate * group[
                                      "cash_borrow_notional"].to_numpy(float)))
        returns = (1.0 + group["gross_return"].to_numpy(float)) * np.exp(-costs) - 1.0
        factor, _ = elapsed_factor(group)
        return annualized_ce(returns, gamma, factor)

    rows = []
    for item in contract["contrasts"]:
        candidate, comparator = align_contrast(panel, item)
        def difference_at(bps: float) -> float:
            return ce_at(candidate, bps) - ce_at(comparator, bps)

        difference_zero = difference_at(0.0)
        difference_max = difference_at(maximum)
        grid = np.linspace(0.0, maximum, 501)
        values = np.asarray([difference_at(float(point)) for point in grid])
        crossing = np.flatnonzero(values[:-1] * values[1:] <= 0)
        if len(crossing) == 0:
            status = ("candidate_positive_throughout_search" if difference_zero > 0
                      else "candidate_negative_throughout_search" if difference_zero < 0
                      else "equal_at_zero_cost_only")
            value = 0.0 if difference_zero == 0 else math.nan
            direction = "none"
        else:
            first = int(crossing[0])
            low, high = float(grid[first]), float(grid[first + 1])
            low_value = difference_at(low)
            for _ in range(60):
                middle = 0.5 * (low + high)
                difference = difference_at(middle)
                if difference == 0:
                    low = high = middle
                    break
                if np.sign(difference) == np.sign(low_value):
                    low = middle
                else:
                    high = middle
            value = 0.5 * (low + high)
            before = difference_at(max(0.0, value - 1e-4))
            after = difference_at(min(maximum, value + 1e-4))
            direction = ("negative_to_positive" if before < after else
                         "positive_to_negative")
            status = "finite_crossing"
        rows.append({
            "contrast_id": item["contrast_id"], "family": item["family"],
            "ce_difference_at_zero_transaction_cost": difference_zero,
            "ce_difference_at_search_limit": difference_max,
            "break_even_transaction_cost_bps": value,
            "break_even_status": status,
            "crossing_direction": direction,
            "annual_short_borrow_percent": 100 * short_rate,
            "annual_cash_borrow_percent": 100 * cash_rate,
            "search_limit_bps": maximum,
        })
    return pd.DataFrame(rows)


def registered_contrast_summary(resampling: pd.DataFrame,
                                influence: pd.DataFrame,
                                break_even: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for contrast_id, group in resampling.groupby("contrast_id", sort=True):
        registered = group[(group["method"] == "moving_block") &
                           (group["block_length"] == 3)]
        stationary = group[(group["method"] == "stationary") &
                           (group["block_length"] == 3)]
        require(len(registered) == 1 and len(stationary) == 1,
                f"Registered contrast specification is missing: {contrast_id}")
        registered = registered.iloc[0]; stationary = stationary.iloc[0]
        leave = influence[influence["contrast_id"] == contrast_id]
        crossing = break_even[break_even["contrast_id"] == contrast_id]
        require(bool(len(leave)) and len(crossing) == 1,
                f"Contrast robustness components are incomplete: {contrast_id}")
        crossing = crossing.iloc[0]
        rows.append({
            "contrast_id": contrast_id, "family": registered["family"],
            "annualized_ce_difference": registered["annualized_ce_difference"],
            "registered_moving_block_3_ci_lower": registered["ci_lower"],
            "registered_moving_block_3_ci_upper": registered["ci_upper"],
            "registered_moving_block_3_holm_p": registered[
                "holm_p_candidate_greater"],
            "stationary_block_3_ci_lower": stationary["ci_lower"],
            "stationary_block_3_ci_upper": stationary["ci_upper"],
            "stationary_block_3_holm_p": stationary["holm_p_candidate_greater"],
            "minimum_holm_p_across_specs": group[
                "holm_p_candidate_greater"].min(),
            "maximum_holm_p_across_specs": group[
                "holm_p_candidate_greater"].max(),
            "specifications_positive_holm_0_05": int(
                group["positive_effect_holm_0_05"].sum()),
            "resampling_specifications": len(group),
            "leave_one_out_min_ce_difference": leave[
                "annualized_ce_difference"].min(),
            "leave_one_out_max_ce_difference": leave[
                "annualized_ce_difference"].max(),
            "leave_one_out_fraction_positive": float((leave[
                "annualized_ce_difference"] > 0).mean()),
            "break_even_transaction_cost_bps": crossing[
                "break_even_transaction_cost_bps"],
            "break_even_status": crossing["break_even_status"],
            "crossing_direction": crossing["crossing_direction"],
            "confirmatory_claim_created": False,
        })
    return pd.DataFrame(rows)


def runtime_manifest(workers: int) -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "), "python_executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "workers": workers, "numpy": np.__version__, "pandas": pd.__version__,
        "blas_environment": {name: os.getenv(name, "") for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }


def execute(release: Path, output: Path, workers: int) -> dict[str, Any]:
    require(not output.exists(), f"Terminal analysis output exists: {output}")
    release_manifest = verify_release(release)
    repo = release.parent.parent
    for relative in FROZEN_CODE:
        live = repo / relative
        frozen = release / "source_snapshot" / relative
        require(live.is_file() and frozen.is_file() and sha256(live) == sha256(frozen),
                f"Live analysis source differs from frozen release: {relative}")
    contract_path = release / "terminal_robustness_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(sha256(contract_path) == release_manifest["contract_sha256"],
            "Frozen terminal contract hash mismatch.")
    workers = max(1, min(int(workers), os.cpu_count() or 1))
    print("[1/7] Loading and auditing six frozen evidence panels...", flush=True)
    panel, ledger = load_panels(release, contract)
    daily_gross = load_daily_gross(release, list(contract["asset_order"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        tables = temporary / "tables"; tables.mkdir()
        print("[2/7] Reconstructing exact daily mark-to-market paths...", flush=True)
        daily, reconciliation = daily_replay(panel, daily_gross, contract)
        print("[3/7] Computing daily downside and tail-risk diagnostics...", flush=True)
        risk = daily_metrics(daily, contract)
        print("[4/7] Re-scoring the frozen-weight friction surface...", flush=True)
        friction = friction_surface(panel, contract)
        print(f"[5/7] Running 50,000-draw resampling grid with {workers} workers...",
              flush=True)
        resampling, influence = resampling_robustness(panel, contract, workers)
        print("[6/7] Computing reality checks, influence, and cost crossings...",
              flush=True)
        reality = reality_checks(panel, contract)
        break_even = break_even_costs(panel, contract)
        contrast_summary = registered_contrast_summary(
            resampling, influence, break_even)
        economics = contract["economics"]
        primary_metrics = friction[
            (friction["scope"] == "complete_periods") &
            (friction["transaction_cost_bps_one_way"] ==
             economics["primary_transaction_cost_bps"]) &
            (friction["annual_short_borrow_percent"] ==
             economics["primary_annual_short_borrow_percent"]) &
            (friction["annual_cash_borrow_percent"] ==
             economics["primary_annual_cash_borrow_percent"])
        ].copy()
        print("[7/7] Writing immutable tables, manifests, and checksums...", flush=True)

        panel.to_csv(tables / "normalized_monthly_evidence_panel.csv", index=False,
                     date_format="%Y-%m-%d")
        ledger.to_csv(tables / "evidence_ledger.csv", index=False)
        daily.to_csv(tables / "daily_strategy_returns.csv", index=False,
                     date_format="%Y-%m-%d")
        reconciliation.to_csv(tables / "daily_monthly_reconciliation.csv", index=False,
                              date_format="%Y-%m-%d")
        risk.to_csv(tables / "daily_tail_risk_metrics.csv", index=False)
        friction.to_csv(tables / "friction_surface.csv", index=False)
        resampling.to_csv(tables / "resampling_robustness.csv", index=False)
        influence.to_csv(tables / "leave_one_period_out.csv", index=False,
                         date_format="%Y-%m-%d")
        reality.to_csv(tables / "white_reality_checks.csv", index=False)
        break_even.to_csv(tables / "break_even_costs.csv", index=False)
        contrast_summary.to_csv(
            tables / "registered_contrast_robustness_summary.csv", index=False)
        primary_metrics.to_csv(
            tables / "primary_economic_metrics.csv", index=False)
        runtime = runtime_manifest(workers)
        (temporary / "runtime_environment.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "status": "terminal_robustness_campaign_complete",
            "analysis_id": contract["analysis_id"],
            "release_contents_sha256": sha256(release / "CONTENTS.sha256"),
            "contract_sha256": sha256(contract_path),
            "source_count": int(len(ledger)),
            "namespaced_strategy_count": int(panel[
                "canonical_strategy_id"].nunique()),
            "window_count": int(panel[["source_id", "window_id"]].drop_duplicates().shape[0]),
            "monthly_rows": int(len(panel)), "daily_rows": int(len(daily)),
            "daily_risk_rows": int(len(risk)), "friction_surface_rows": int(len(friction)),
            "resampling_rows": int(len(resampling)),
            "registered_contrast_summary_rows": int(len(contrast_summary)),
            "primary_economic_metric_rows": int(len(primary_metrics)),
            "leave_one_period_out_rows": int(len(influence)),
            "all_daily_paths_reconciled": bool(reconciliation[
                "reconciliation_pass"].all()),
            "maximum_absolute_gross_reconciliation_error": float(
                reconciliation["gross_error"].abs().max()),
            "maximum_absolute_net_reconciliation_error": float(
                reconciliation["net_error"].abs().max()),
            "policy_retraining_performed": False,
            "model_selection_performed": False,
            "confirmatory_claim_created": False,
            "evidence_classes_kept_separate": True,
            "stop_rule": contract["stop_rule"],
        }
        (temporary / "terminal_robustness_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_contents(temporary)
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=max(1, (os.cpu_count() or 2) - 2), type=int)
    args = parser.parse_args()
    try:
        result = execute(args.release.resolve(), args.output.resolve(), args.workers)
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            TerminalProtocolError, TerminalAnalysisError) as error:
        print(f"TERMINAL ROBUSTNESS ANALYSIS FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
