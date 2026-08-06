#!/usr/bin/env python3
"""Post-holdout explanatory audit of a completed locked evaluation batch.

This module is deliberately separate from ``publication_pipeline.py``.  It
does not alter, re-run, or select the locked primary evaluation.  It explains
how a predeclared multi-seed arithmetic ensemble behaves after holdout access.
All outputs are labelled post-holdout explanatory and are unsuitable for new
confirmatory superiority claims.
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
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


CLASSIFICATION = "post_holdout_explanatory"
NOTICE = "POST-HOLDOUT EXPLANATORY — NOT CONFIRMATORY"
KEYS = ["window_id", "decision_date", "holding_end_date"]


class ExplanatoryAnalysisError(RuntimeError):
    """Raised when a completed-batch or explanatory-analysis invariant fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ExplanatoryAnalysisError(f"{field} must be boolean; received {value!r}.")


@dataclass(frozen=True)
class AnalysisContract:
    raw: dict[str, Any]

    @classmethod
    def read(cls, path: Path) -> "AnalysisContract":
        raw = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "schema_version", "analysis_classification",
            "confirmatory_use_permitted", "expected_seed_count",
            "seed_strategy_prefix", "ensemble_strategy_id", "sample_scope",
            "periods_per_year", "initial_wealth", "arithmetic_tolerance",
            "k_seed_sizes", "bootstrap_replications", "analysis_seed",
        }
        missing = sorted(required.difference(raw))
        if missing:
            raise ExplanatoryAnalysisError(
                f"Analysis contract is missing fields: {', '.join(missing)}")
        if int(raw["schema_version"]) != 1:
            raise ExplanatoryAnalysisError("Only analysis schema_version=1 is supported.")
        if raw["analysis_classification"] != CLASSIFICATION:
            raise ExplanatoryAnalysisError(
                f"analysis_classification must be {CLASSIFICATION!r}.")
        if parse_bool(raw["confirmatory_use_permitted"], "confirmatory_use_permitted"):
            raise ExplanatoryAnalysisError(
                "This module cannot produce confirmatory outputs after holdout access.")
        if int(raw["expected_seed_count"]) < 2:
            raise ExplanatoryAnalysisError("expected_seed_count must be at least two.")
        if raw["sample_scope"] not in {"complete_periods", "locked_all"}:
            raise ExplanatoryAnalysisError(
                "sample_scope must be complete_periods or locked_all.")
        if float(raw["periods_per_year"]) <= 0 or float(raw["initial_wealth"]) <= 0:
            raise ExplanatoryAnalysisError("Annualisation and wealth inputs must be positive.")
        if not 0 < float(raw["arithmetic_tolerance"]) < 1e-2:
            raise ExplanatoryAnalysisError("arithmetic_tolerance must lie in (0, 0.01).")
        sizes = [int(value) for value in raw["k_seed_sizes"]]
        if not sizes or sizes != sorted(set(sizes)) or min(sizes) < 1:
            raise ExplanatoryAnalysisError(
                "k_seed_sizes must be a nonempty, sorted list of unique positive integers.")
        if max(sizes) > int(raw["expected_seed_count"]):
            raise ExplanatoryAnalysisError("k_seed_sizes cannot exceed expected_seed_count.")
        if int(raw["bootstrap_replications"]) < 100:
            raise ExplanatoryAnalysisError(
                "At least 100 deterministic seed-bootstrap replications are required.")
        return cls(raw=raw)

    def __getitem__(self, name: str) -> Any:
        return self.raw[name]


@dataclass
class BatchData:
    root: Path
    locked_manifest: dict[str, Any]
    publication_manifest: dict[str, Any]
    economics: dict[str, Any]
    strategy_manifest: pd.DataFrame
    scored: pd.DataFrame
    realized: pd.DataFrame
    assets: list[str]
    seed_ids: list[str]
    seed_numbers: list[int]
    seed_weights: np.ndarray
    ensemble_weights: np.ndarray
    verification: pd.DataFrame


def labelled(frame: pd.DataFrame, explanatory_kind: str) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "analysis_classification", CLASSIFICATION)
    output.insert(1, "confirmatory_use_permitted", False)
    output.insert(2, "explanatory_kind", explanatory_kind)
    return output


def _read_csv_dates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for name in ["decision_date", "holding_end_date"]:
        if name in frame:
            frame[name] = pd.to_datetime(frame[name], errors="raise")
    return frame


def _aligned_strategy(frame: pd.DataFrame, strategy_id: str, reference: pd.DataFrame) -> pd.DataFrame:
    group = frame[frame["strategy_id"] == strategy_id].sort_values(
        KEYS, kind="stable").reset_index(drop=True)
    if len(group) != len(reference) or not np.array_equal(
        group[KEYS].astype(str).to_numpy(), reference[KEYS].astype(str).to_numpy()
    ):
        raise ExplanatoryAnalysisError(
            f"Strategy {strategy_id} is not one-to-one aligned to the locked panel.")
    return group


def load_completed_batch(batch: Path, contract: AnalysisContract) -> BatchData:
    batch = batch.resolve()
    required = {
        "locked_manifest": batch / "locked_batch_manifest.json",
        "publication_manifest": batch / "publication_results" / "run_manifest.json",
        "economics": batch / "benchmark_weights" / "benchmark_contract.json",
        "scored": batch / "publication_results" / "raw" / "scored_monthly_panel.csv",
        "realized": batch / "inputs" / "realized_asset_gross.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ExplanatoryAnalysisError(
            f"Completed locked batch is missing: {', '.join(missing)}")
    root_manifest_path = batch / "strategy_manifest.csv"
    validated_manifest_path = (
        batch / "publication_results" / "raw" / "validated_strategy_manifest.csv")
    manifest_path = validated_manifest_path if validated_manifest_path.is_file() else root_manifest_path
    if not manifest_path.is_file() or not root_manifest_path.is_file():
        raise ExplanatoryAnalysisError(
            "Both the frozen root strategy manifest and a readable strategy manifest are required.")

    locked = json.loads(required["locked_manifest"].read_text(encoding="utf-8"))
    publication = json.loads(required["publication_manifest"].read_text(encoding="utf-8"))
    economics = json.loads(required["economics"].read_text(encoding="utf-8"))
    if locked.get("status") != "complete" or not parse_bool(
        locked.get("holdout_accessed", False), "holdout_accessed"):
        raise ExplanatoryAnalysisError(
            "Analysis is permitted only for a complete batch whose holdout was accessed.")
    expected = int(contract["expected_seed_count"])
    if int(locked.get("full_policy_count", -1)) != expected:
        raise ExplanatoryAnalysisError(
            f"Locked batch full_policy_count is not the required {expected}.")
    ensemble_id = str(contract["ensemble_strategy_id"])
    if publication.get("primary_strategy_id") != ensemble_id:
        raise ExplanatoryAnalysisError(
            "The explanatory ensemble is not the frozen publication primary strategy.")

    scored = _read_csv_dates(required["scored"])
    realized = _read_csv_dates(required["realized"])
    if "is_complete_period" not in scored or "is_complete_period" not in realized:
        raise ExplanatoryAnalysisError("Locked panels must identify complete periods.")
    scored["is_complete_period"] = scored["is_complete_period"].map(
        lambda x: parse_bool(x, "is_complete_period"))
    realized["is_complete_period"] = realized["is_complete_period"].map(
        lambda x: parse_bool(x, "is_complete_period"))
    assets = [name[2:] for name in realized if name.startswith("g_")]
    weight_columns = [f"w_{asset}" for asset in assets]
    if not assets or any(name not in scored for name in weight_columns):
        raise ExplanatoryAnalysisError("Asset gross-return and scored-weight columns disagree.")
    if realized.duplicated(KEYS).any():
        raise ExplanatoryAnalysisError("Realized panel contains duplicate period keys.")
    reference = realized.sort_values(KEYS, kind="stable").reset_index(drop=True)

    manifest = pd.read_csv(manifest_path)
    root_manifest = pd.read_csv(root_manifest_path)
    prefix = str(contract["seed_strategy_prefix"])
    seed_rows = manifest[manifest["strategy_id"].astype(str).str.startswith(prefix)].copy()
    if len(seed_rows) != expected:
        raise ExplanatoryAnalysisError(
            f"Expected {expected} seed strategies; found {len(seed_rows)}.")
    if seed_rows["strategy_id"].duplicated().any() or seed_rows["seed"].duplicated().any():
        raise ExplanatoryAnalysisError("Seed identifiers and numerical seeds must be unique.")
    seed_rows["seed"] = seed_rows["seed"].astype(int)
    seed_rows = seed_rows.sort_values("seed", kind="stable")
    seed_ids = seed_rows["strategy_id"].astype(str).tolist()
    seed_numbers = seed_rows["seed"].astype(int).tolist()
    if ensemble_id not in set(scored["strategy_id"]):
        raise ExplanatoryAnalysisError("Frozen arithmetic ensemble is absent from the scored panel.")

    frozen_rows = root_manifest.set_index("strategy_id").reindex(seed_ids)
    if frozen_rows.isna().all(axis=1).any():
        raise ExplanatoryAnalysisError("A seed is absent from the frozen root manifest.")
    tolerance = float(contract["arithmetic_tolerance"])
    arrays: list[np.ndarray] = []
    verification_rows: list[dict[str, Any]] = []
    paths_seen: list[str] = []
    hashes_seen: list[str] = []
    checkpoint_hashes: list[str] = []
    for strategy_id, seed in zip(seed_ids, seed_numbers):
        panel = _aligned_strategy(scored, strategy_id, reference)
        row = frozen_rows.loc[strategy_id]
        relative = str(row["weight_log_path"])
        path = batch / relative
        if not path.is_file():
            raise ExplanatoryAnalysisError(f"Frozen seed weight log is missing: {relative}")
        actual_hash = sha256_file(path)
        recorded_hash = str(row["weight_log_sha256"]).lower()
        if actual_hash != recorded_hash:
            raise ExplanatoryAnalysisError(f"Weight hash mismatch for {strategy_id}.")
        weights = _read_csv_dates(path).sort_values(KEYS, kind="stable").reset_index(drop=True)
        if len(weights) != len(reference) or not np.array_equal(
            weights[KEYS].astype(str).to_numpy(), reference[KEYS].astype(str).to_numpy()
        ):
            raise ExplanatoryAnalysisError(f"Frozen weight log is misaligned for {strategy_id}.")
        values = weights[weight_columns].to_numpy(float)
        if not np.isfinite(values).all():
            raise ExplanatoryAnalysisError(f"Non-finite weights for {strategy_id}.")
        panel_error = float(np.max(np.abs(values - panel[weight_columns].to_numpy(float))))
        if panel_error > tolerance:
            raise ExplanatoryAnalysisError(
                f"Frozen weight log and scored panel disagree for {strategy_id}.")
        arrays.append(values)
        paths_seen.append(relative)
        hashes_seen.append(actual_hash)
        checkpoint_hashes.append(str(row.get("checkpoint_sha256", "")))
        verification_rows.append({
            "strategy_id": strategy_id,
            "seed": seed,
            "weight_log_path": relative,
            "weight_log_sha256": actual_hash,
            "checkpoint_path": str(row.get("checkpoint_path", "")),
            "checkpoint_sha256": str(row.get("checkpoint_sha256", "")),
            "config_sha256": str(row.get("config_sha256", "")),
            "code_sha256": str(row.get("code_sha256", "")),
            "maximum_panel_weight_error": panel_error,
        })
    if len(set(paths_seen)) != expected or len(set(hashes_seen)) != expected:
        raise ExplanatoryAnalysisError("Seed weight paths and content hashes must be unique.")
    applicable_checkpoint_hashes = [
        value for value in checkpoint_hashes if value and value != "not_applicable"]
    if len(applicable_checkpoint_hashes) != expected or len(set(applicable_checkpoint_hashes)) != expected:
        raise ExplanatoryAnalysisError("Seed checkpoint hashes must be present and unique.")

    seed_weights = np.stack(arrays)
    ensemble_panel = _aligned_strategy(scored, ensemble_id, reference)
    ensemble_weights = ensemble_panel[weight_columns].to_numpy(float)
    arithmetic = seed_weights.mean(axis=0)
    ensemble_error = float(np.max(np.abs(arithmetic - ensemble_weights)))
    if ensemble_error > tolerance:
        raise ExplanatoryAnalysisError(
            f"Frozen ensemble is not the exact arithmetic mean; max error={ensemble_error:.3g}.")
    gross_average_error = float(np.max(np.abs(
        np.stack([
            _aligned_strategy(scored, strategy_id, reference)["gross_return"].to_numpy(float)
            for strategy_id in seed_ids
        ]).mean(axis=0) - ensemble_panel["gross_return"].to_numpy(float)
    )))
    if gross_average_error > tolerance:
        raise ExplanatoryAnalysisError(
            "Ensemble gross returns are not the exact mean of seed gross returns.")

    verification = pd.DataFrame(verification_rows)
    verification["all_weight_paths_unique"] = len(set(paths_seen)) == expected
    verification["all_weight_hashes_unique"] = len(set(hashes_seen)) == expected
    verification["all_checkpoint_hashes_unique"] = len(set(applicable_checkpoint_hashes)) == expected
    verification["maximum_ensemble_weight_error"] = ensemble_error
    verification["maximum_ensemble_gross_average_error"] = gross_average_error
    verification["all_config_hashes_identical_across_seeds"] = (
        verification["config_sha256"].nunique(dropna=False) == 1)
    verification["all_code_hashes_identical_across_seeds"] = (
        verification["code_sha256"].nunique(dropna=False) == 1)
    return BatchData(
        root=batch, locked_manifest=locked, publication_manifest=publication,
        economics=economics, strategy_manifest=manifest, scored=scored,
        realized=reference, assets=assets, seed_ids=seed_ids,
        seed_numbers=seed_numbers, seed_weights=seed_weights,
        ensemble_weights=ensemble_weights, verification=verification,
    )


def scope_rows(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "complete_periods":
        return frame[frame["is_complete_period"]].copy()
    return frame.copy()


def crra_utility(returns: np.ndarray, gamma: float) -> np.ndarray:
    gross = 1.0 + np.asarray(returns, dtype=float)
    if np.any(gross <= 0):
        raise ExplanatoryAnalysisError("CRRA utility is undefined for nonpositive gross wealth.")
    if abs(gamma - 1.0) < 1e-12:
        return np.log(gross)
    return (np.power(gross, 1.0 - gamma) - 1.0) / (1.0 - gamma)


def certainty_equivalent(mean_utility: float, gamma: float) -> float:
    if abs(gamma - 1.0) < 1e-12:
        return math.exp(mean_utility) - 1.0
    base = 1.0 + (1.0 - gamma) * mean_utility
    if base <= 0:
        return math.nan
    return math.pow(base, 1.0 / (1.0 - gamma)) - 1.0


def empirical_metrics(frame: pd.DataFrame, contract: AnalysisContract, gamma: float) -> dict[str, float]:
    returns = frame["net_return"].to_numpy(float)
    if len(returns) < 2 or not np.isfinite(returns).all():
        raise ExplanatoryAnalysisError("At least two finite returns are required.")
    periods = float(contract["periods_per_year"])
    multiple = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.r_[1.0, multiple])[1:]
    drawdowns = 1.0 - multiple / peaks
    annual_mean = periods * float(np.mean(returns))
    annual_vol = math.sqrt(periods) * float(np.std(returns, ddof=1))
    threshold = float(np.quantile(returns, 0.05))
    tail = returns[returns <= threshold]
    mean_utility = float(np.mean(crra_utility(returns, gamma)))
    monthly_ce = certainty_equivalent(mean_utility, gamma)
    return {
        "observations": float(len(returns)),
        "total_return": float(multiple[-1] - 1.0),
        "terminal_wealth": float(contract["initial_wealth"]) * float(multiple[-1]),
        "cagr": float(math.pow(multiple[-1], periods / len(returns)) - 1.0),
        "annual_arithmetic_return": annual_mean,
        "annual_volatility": annual_vol,
        "sharpe_ratio": annual_mean / annual_vol if annual_vol > 0 else math.nan,
        "max_drawdown": float(np.max(drawdowns)),
        "realized_cvar05_loss": float(-np.mean(tail)),
        "annualized_certainty_equivalent_return": (
            float(math.pow(1.0 + monthly_ce, periods) - 1.0)
            if math.isfinite(monthly_ce) else math.nan),
        "positive_month_fraction": float(np.mean(returns > 0)),
        "mean_monthly_turnover": float(frame["turnover"].mean()),
        "mean_gross_exposure": float(frame["gross_exposure"].mean()),
        "mean_short_notional": float(frame["short_notional"].mean()),
        "mean_transaction_cost_bps": float(frame["transaction_cost"].mean() * 10000.0),
        "mean_financing_cost_bps": float(frame["financing_cost"].mean() * 10000.0),
    }


def build_metrics(data: BatchData, contract: AnalysisContract) -> pd.DataFrame:
    scoped = scope_rows(data.scored, str(contract["sample_scope"]))
    gamma = float(data.economics["crra_gamma"])
    ids = data.seed_ids + [str(contract["ensemble_strategy_id"])]
    benchmark_ids = data.strategy_manifest.loc[
        data.strategy_manifest["role"].astype(str) == "benchmark", "strategy_id"
    ].astype(str).tolist()
    rows = []
    for strategy_id in ids + benchmark_ids:
        group = scoped[scoped["strategy_id"] == strategy_id].sort_values(KEYS)
        if group.empty:
            continue
        row = {"strategy_id": strategy_id}
        row.update(empirical_metrics(group, contract, gamma))
        rows.append(row)
    return pd.DataFrame(rows)


def seed_dispersion_tables(
    metrics: pd.DataFrame, data: BatchData, contract: AnalysisContract
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seeds = metrics[metrics["strategy_id"].isin(data.seed_ids)].set_index("strategy_id")
    ensemble_id = str(contract["ensemble_strategy_id"])
    ensemble = metrics.set_index("strategy_id").loc[ensemble_id]
    report_metrics = [
        "cagr", "annual_volatility", "sharpe_ratio", "max_drawdown",
        "realized_cvar05_loss", "annualized_certainty_equivalent_return",
        "mean_monthly_turnover", "mean_gross_exposure", "mean_short_notional",
        "terminal_wealth",
    ]
    lower_is_better = {
        "annual_volatility", "max_drawdown", "realized_cvar05_loss",
        "mean_monthly_turnover",
    }
    dispersion_rows = []
    comparison_rows = []
    for metric in report_metrics:
        values = seeds[metric].astype(float)
        dispersion_rows.append({
            "metric": metric,
            "n_seeds": len(values),
            "minimum": values.min(),
            "q05": values.quantile(0.05),
            "median": values.median(),
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "q95": values.quantile(0.95),
            "maximum": values.max(),
            "seed_at_minimum": values.idxmin(),
            "seed_at_maximum": values.idxmax(),
        })
        value = float(ensemble[metric])
        better = int((values < value).sum()) if metric in lower_is_better else int((values > value).sum())
        comparison_rows.append({
            "metric": metric,
            "preferred_direction": "lower" if metric in lower_is_better else "higher",
            "ensemble_value": value,
            "seed_minimum": values.min(),
            "seed_median": values.median(),
            "seed_mean": values.mean(),
            "seed_maximum": values.max(),
            "seeds_better_than_ensemble": better,
            "ensemble_rank_among_seeds_plus_ensemble": better + 1,
        })

    benchmark_ids = data.strategy_manifest.loc[
        data.strategy_manifest["role"].astype(str) == "benchmark", "strategy_id"
    ].astype(str).tolist()
    benchmark_metrics = metrics.set_index("strategy_id").reindex(benchmark_ids)
    win_rows = []
    for benchmark_id, benchmark in benchmark_metrics.iterrows():
        if benchmark.isna().all():
            continue
        for metric in [
            "cagr", "sharpe_ratio", "annualized_certainty_equivalent_return",
            "max_drawdown", "realized_cvar05_loss",
        ]:
            lower = metric in lower_is_better
            count = int((seeds[metric] < benchmark[metric]).sum()) if lower else int(
                (seeds[metric] > benchmark[metric]).sum())
            win_rows.append({
                "benchmark_id": benchmark_id,
                "metric": metric,
                "preferred_direction": "lower" if lower else "higher",
                "benchmark_value": float(benchmark[metric]),
                "outperforming_seed_count": count,
                "seed_count": len(seeds),
                "outperforming_seed_fraction": count / len(seeds),
            })
    return (
        labelled(pd.DataFrame(dispersion_rows), "seed_metric_dispersion"),
        labelled(pd.DataFrame(comparison_rows), "ensemble_vs_seed_distribution"),
        labelled(pd.DataFrame(win_rows), "seed_benchmark_win_fraction"),
    )


def correlation_and_diversity_tables(
    data: BatchData, contract: AnalysisContract
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scoped = scope_rows(data.scored, str(contract["sample_scope"]))
    returns = scoped[scoped["strategy_id"].isin(data.seed_ids)].pivot(
        index=KEYS, columns="strategy_id", values="net_return").reindex(columns=data.seed_ids)
    return_corr = returns.corr()
    mask = data.realized["is_complete_period"].to_numpy(bool)
    if contract["sample_scope"] == "locked_all":
        mask = np.ones(len(data.realized), dtype=bool)
    weights = data.seed_weights[:, mask, :]
    flat = weights.reshape(len(data.seed_ids), -1)
    weight_corr = np.corrcoef(flat)
    return_values = return_corr.to_numpy()[np.triu_indices(len(data.seed_ids), 1)]
    weight_values = weight_corr[np.triu_indices(len(data.seed_ids), 1)]
    summaries = []
    for name, values in [
        ("pairwise_monthly_net_return_correlation", return_values),
        ("pairwise_flattened_weight_path_correlation", weight_values),
    ]:
        summaries.append({
            "measure": name,
            "pairs": len(values),
            "minimum": np.min(values),
            "q05": np.quantile(values, 0.05),
            "median": np.median(values),
            "mean": np.mean(values),
            "q95": np.quantile(values, 0.95),
            "maximum": np.max(values),
        })
    positive = weights > 0
    negative = weights < 0
    mixed = positive.any(axis=0) & negative.any(axis=0)
    minority = np.minimum(positive.mean(axis=0), negative.mean(axis=0))
    diversity_rows = [{
        "asset": "ALL",
        "month_asset_cells": mixed.size,
        "mixed_sign_cells": int(mixed.sum()),
        "mixed_sign_cell_fraction": float(mixed.mean()),
        "mean_minority_sign_share": float(minority.mean()),
        "mean_cross_seed_weight_std": float(weights.std(axis=0, ddof=1).mean()),
    }]
    for index, asset in enumerate(data.assets):
        diversity_rows.append({
            "asset": asset,
            "month_asset_cells": mixed[:, index].size,
            "mixed_sign_cells": int(mixed[:, index].sum()),
            "mixed_sign_cell_fraction": float(mixed[:, index].mean()),
            "mean_minority_sign_share": float(minority[:, index].mean()),
            "mean_cross_seed_weight_std": float(
                weights[:, :, index].std(axis=0, ddof=1).mean()),
        })
    return_frame = pd.DataFrame(return_corr, index=data.seed_ids, columns=data.seed_ids)
    return_frame.insert(0, "strategy_id", data.seed_ids)
    weight_frame = pd.DataFrame(weight_corr, index=data.seed_ids, columns=data.seed_ids)
    weight_frame.insert(0, "strategy_id", data.seed_ids)
    return (
        labelled(pd.DataFrame(summaries), "pairwise_correlation_summary"),
        labelled(return_frame.reset_index(drop=True), "pairwise_return_correlation_matrix"),
        labelled(weight_frame.reset_index(drop=True), "pairwise_weight_correlation_matrix"),
        labelled(pd.DataFrame(diversity_rows), "weight_sign_and_dispersion"),
    )


def mechanism_tables(
    data: BatchData, metrics: pd.DataFrame, contract: AnalysisContract
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoped = scope_rows(data.scored, str(contract["sample_scope"]))
    seed = scoped[scoped["strategy_id"].isin(data.seed_ids)]
    ensemble_id = str(contract["ensemble_strategy_id"])
    ensemble = scoped[scoped["strategy_id"] == ensemble_id].sort_values(KEYS)
    averages = seed.groupby(KEYS, sort=False).agg(
        mean_seed_gross_return=("gross_return", "mean"),
        mean_seed_net_return=("net_return", "mean"),
        mean_seed_gross_exposure=("gross_exposure", "mean"),
        mean_seed_short_notional=("short_notional", "mean"),
        mean_seed_turnover=("turnover", "mean"),
        mean_seed_transaction_cost=("transaction_cost", "mean"),
        mean_seed_financing_cost=("financing_cost", "mean"),
    ).reset_index().sort_values(KEYS)
    fields = KEYS + [
        "gross_return", "net_return", "gross_exposure", "short_notional",
        "turnover", "transaction_cost", "financing_cost",
    ]
    joined = averages.merge(
        ensemble[fields], on=KEYS, how="inner", validate="one_to_one",
        suffixes=("", "_ensemble"))
    joined = joined.rename(columns={
        "gross_return": "ensemble_gross_return",
        "net_return": "ensemble_net_return",
        "gross_exposure": "ensemble_gross_exposure",
        "short_notional": "ensemble_short_notional",
        "turnover": "ensemble_turnover",
        "transaction_cost": "ensemble_transaction_cost",
        "financing_cost": "ensemble_financing_cost",
    })
    net_exposure = float(data.economics["net_exposure"])
    extra_seed = joined["mean_seed_gross_exposure"] - abs(net_exposure)
    extra_ensemble = joined["ensemble_gross_exposure"] - abs(net_exposure)
    joined["incremental_gross_cancellation_fraction"] = np.where(
        extra_seed > 1e-12, 1.0 - extra_ensemble / extra_seed, np.nan)
    joined["short_notional_cancellation_fraction"] = np.where(
        joined["mean_seed_short_notional"] > 1e-12,
        1.0 - joined["ensemble_short_notional"] / joined["mean_seed_short_notional"],
        np.nan)
    joined["gross_return_averaging_error"] = (
        joined["ensemble_gross_return"] - joined["mean_seed_gross_return"])
    joined["transaction_cost_saving_bps"] = 10000.0 * (
        joined["mean_seed_transaction_cost"] - joined["ensemble_transaction_cost"])
    joined["financing_cost_saving_bps"] = 10000.0 * (
        joined["mean_seed_financing_cost"] - joined["ensemble_financing_cost"])
    joined["net_return_advantage_bps"] = 10000.0 * (
        joined["ensemble_net_return"] - joined["mean_seed_net_return"])

    seed_metrics = metrics[metrics["strategy_id"].isin(data.seed_ids)]
    ensemble_metrics = metrics.set_index("strategy_id").loc[ensemble_id]
    mean_seed_net_terminal = float(contract["initial_wealth"]) * float(np.prod(
        1.0 + joined["mean_seed_net_return"].to_numpy(float)))
    incremental_seed_gross = (
        joined["mean_seed_gross_exposure"].mean() - abs(net_exposure))
    mean_seed_short = joined["mean_seed_short_notional"].mean()
    incremental_gross_cancellation = (
        1.0 - (
            joined["ensemble_gross_exposure"].mean() - abs(net_exposure)
        ) / incremental_seed_gross
        if incremental_seed_gross > 1e-12 else np.nan
    )
    short_cancellation = (
        1.0 - joined["ensemble_short_notional"].mean() / mean_seed_short
        if mean_seed_short > 1e-12 else np.nan
    )
    summary = pd.DataFrame([{
        "seed_count": len(data.seed_ids),
        "mean_seed_gross_exposure": joined["mean_seed_gross_exposure"].mean(),
        "ensemble_gross_exposure": joined["ensemble_gross_exposure"].mean(),
        "incremental_gross_cancellation_fraction": incremental_gross_cancellation,
        "mean_seed_short_notional": joined["mean_seed_short_notional"].mean(),
        "ensemble_short_notional": joined["ensemble_short_notional"].mean(),
        "short_notional_cancellation_fraction": short_cancellation,
        "mean_seed_turnover": joined["mean_seed_turnover"].mean(),
        "ensemble_turnover": joined["ensemble_turnover"].mean(),
        "transaction_cost_saving_bps_per_period": joined["transaction_cost_saving_bps"].mean(),
        "financing_cost_saving_bps_per_period": joined["financing_cost_saving_bps"].mean(),
        "net_return_advantage_bps_per_period": joined["net_return_advantage_bps"].mean(),
        "maximum_gross_return_averaging_error": joined["gross_return_averaging_error"].abs().max(),
        "mean_individual_seed_terminal_wealth": seed_metrics["terminal_wealth"].mean(),
        "terminal_wealth_of_mean_seed_net_return_path": mean_seed_net_terminal,
        "ensemble_terminal_wealth": ensemble_metrics["terminal_wealth"],
    }])
    return (
        labelled(joined, "ensemble_mechanism_by_period"),
        labelled(summary, "ensemble_mechanism_summary"),
    )


def score_weight_path(
    weights: np.ndarray,
    realized: pd.DataFrame,
    assets: list[str],
    economics: dict[str, Any],
    contract: AnalysisContract,
    turnover_mode: str = "target_to_target",
) -> pd.DataFrame:
    if turnover_mode not in {"target_to_target", "drift_aware"}:
        raise ExplanatoryAnalysisError(f"Unsupported turnover mode {turnover_mode!r}.")
    weight_columns = [f"w_{asset}" for asset in assets]
    gross_columns = [f"g_{asset}" for asset in assets]
    if weights.shape != (len(realized), len(assets)):
        raise ExplanatoryAnalysisError("Weight tensor has the wrong locked-panel shape.")
    rows: list[dict[str, Any]] = []
    periods = float(contract["periods_per_year"])
    for window_id, index in realized.groupby("window_id", sort=False).groups.items():
        positions = list(index)
        previous = np.repeat(float(economics["net_exposure"]) / len(assets), len(assets))
        for position in positions:
            record = realized.loc[position]
            w = np.asarray(weights[position], dtype=float)
            g = record[gross_columns].to_numpy(float)
            turnover = float(np.abs(w - previous).sum())
            short_notional = float(np.maximum(-w, 0.0).sum())
            cash_borrow = max(float(w.sum()) - 1.0, 0.0)
            transaction_cost = float(economics["turnover_cost"]) * turnover
            financing_cost = (
                float(economics["annual_short_borrow_rate"]) * short_notional
                + float(economics["annual_cash_borrow_rate"]) * cash_borrow
            ) / periods
            gross_portfolio = 1.0 + float(np.dot(w, g - 1.0))
            net_gross = gross_portfolio * math.exp(-transaction_cost - financing_cost)
            if not math.isfinite(net_gross) or net_gross <= 0:
                raise ExplanatoryAnalysisError(
                    f"Exploratory ensemble is insolvent in window {window_id}.")
            row = {name: record[name] for name in KEYS}
            row.update({
                "is_complete_period": bool(record["is_complete_period"]),
                "gross_return": gross_portfolio - 1.0,
                "net_return": net_gross - 1.0,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "financing_cost": financing_cost,
                "short_notional": short_notional,
                "gross_exposure": float(np.abs(w).sum()),
            })
            row.update(dict(zip(weight_columns, w)))
            rows.append(row)
            if turnover_mode == "drift_aware":
                previous = w * g / gross_portfolio
            else:
                previous = w
    return pd.DataFrame(rows)


def k_seed_sensitivity(
    data: BatchData, contract: AnalysisContract
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gamma = float(data.economics["crra_gamma"])
    rows = []
    deterministic_rows = []
    rng = np.random.default_rng(int(contract["analysis_seed"]))
    replications = int(contract["bootstrap_replications"])
    sizes = [int(value) for value in contract["k_seed_sizes"]]
    for size in sizes:
        prefix_indices = np.arange(size)
        prefix_weights = data.seed_weights[prefix_indices].mean(axis=0)
        prefix_score = scope_rows(score_weight_path(
            prefix_weights, data.realized, data.assets, data.economics, contract),
            str(contract["sample_scope"]))
        prefix_metric = empirical_metrics(prefix_score, contract, gamma)
        deterministic_rows.append({
            "k": size,
            "selection_rule": "sorted_seed_prefix",
            "member_seed_ids": "|".join(data.seed_ids[:size]),
            **prefix_metric,
        })
        for replication in range(replications):
            sampled = rng.integers(0, len(data.seed_ids), size=size)
            ensemble = data.seed_weights[sampled].mean(axis=0)
            scored = scope_rows(score_weight_path(
                ensemble, data.realized, data.assets, data.economics, contract),
                str(contract["sample_scope"]))
            metric = empirical_metrics(scored, contract, gamma)
            rows.append({
                "k": size,
                "replication": replication + 1,
                "sampling_rule": "seed_bootstrap_with_replacement",
                "member_seed_ids": "|".join(data.seed_ids[index] for index in sampled),
                **metric,
            })
    draws = pd.DataFrame(rows)
    deterministic = pd.DataFrame(deterministic_rows)
    summary_rows = []
    for size, group in draws.groupby("k", sort=True):
        deterministic_row = deterministic[deterministic["k"] == size].iloc[0]
        row = {
            "k": size,
            "bootstrap_replications": len(group),
            "bootstrap_sampling_rule": "seed_bootstrap_with_replacement",
            "deterministic_rule": "sorted_seed_prefix",
        }
        for metric in [
            "cagr", "sharpe_ratio", "max_drawdown", "mean_monthly_turnover",
            "mean_gross_exposure", "mean_short_notional", "terminal_wealth",
        ]:
            row[f"{metric}_bootstrap_q05"] = group[metric].quantile(0.05)
            row[f"{metric}_bootstrap_median"] = group[metric].median()
            row[f"{metric}_bootstrap_q95"] = group[metric].quantile(0.95)
            row[f"{metric}_deterministic_prefix"] = deterministic_row[metric]
        summary_rows.append(row)
    return (
        labelled(draws, "exploratory_k_seed_bootstrap_draw"),
        labelled(deterministic, "exploratory_k_seed_deterministic_prefix"),
        labelled(pd.DataFrame(summary_rows), "exploratory_k_seed_summary"),
    )


def drift_turnover_sensitivity(
    data: BatchData, contract: AnalysisContract
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gamma = float(data.economics["crra_gamma"])
    ids = data.seed_ids + [str(contract["ensemble_strategy_id"])]
    arrays = list(data.seed_weights) + [data.ensemble_weights]
    reported = scope_rows(data.scored, str(contract["sample_scope"]))
    rows = []
    tolerance = float(contract["arithmetic_tolerance"])
    for strategy_id, weights in zip(ids, arrays):
        target_score_all = score_weight_path(
            weights, data.realized, data.assets, data.economics, contract,
            turnover_mode="target_to_target")
        drift_score_all = score_weight_path(
            weights, data.realized, data.assets, data.economics, contract,
            turnover_mode="drift_aware")
        target = scope_rows(target_score_all, str(contract["sample_scope"]))
        drift = scope_rows(drift_score_all, str(contract["sample_scope"]))
        frozen = reported[reported["strategy_id"] == strategy_id].sort_values(KEYS)
        target = target.sort_values(KEYS)
        validation_error = float(np.max(np.abs(
            target["net_return"].to_numpy(float) - frozen["net_return"].to_numpy(float))))
        if validation_error > tolerance:
            raise ExplanatoryAnalysisError(
                f"Target-to-target rescoring does not reproduce {strategy_id}.")
        target_metric = empirical_metrics(target, contract, gamma)
        drift_metric = empirical_metrics(drift, contract, gamma)
        rows.append({
            "strategy_id": strategy_id,
            "role": "ensemble" if strategy_id == contract["ensemble_strategy_id"] else "seed",
            "target_to_target_validation_error": validation_error,
            "reported_turnover": target_metric["mean_monthly_turnover"],
            "drift_aware_turnover": drift_metric["mean_monthly_turnover"],
            "turnover_difference": (
                drift_metric["mean_monthly_turnover"] - target_metric["mean_monthly_turnover"]),
            "reported_transaction_cost_bps": target_metric["mean_transaction_cost_bps"],
            "drift_aware_transaction_cost_bps": drift_metric["mean_transaction_cost_bps"],
            "reported_cagr": target_metric["cagr"],
            "drift_aware_cagr": drift_metric["cagr"],
            "reported_sharpe_ratio": target_metric["sharpe_ratio"],
            "drift_aware_sharpe_ratio": drift_metric["sharpe_ratio"],
            "reported_terminal_wealth": target_metric["terminal_wealth"],
            "drift_aware_terminal_wealth": drift_metric["terminal_wealth"],
        })
    details = pd.DataFrame(rows)
    seed_rows = details[details["role"] == "seed"]
    ensemble_row = details[details["role"] == "ensemble"].iloc[0]
    summary = pd.DataFrame([
        {
            "group": "mean_individual_seed",
            **{name: seed_rows[name].mean() for name in details if name not in {"strategy_id", "role"}},
        },
        {
            "group": "arithmetic_ensemble",
            **{name: ensemble_row[name] for name in details if name not in {"strategy_id", "role"}},
        },
    ])
    return (
        labelled(details, "exploratory_drift_aware_turnover_by_strategy"),
        labelled(summary, "exploratory_drift_aware_turnover_summary"),
    )


def save_figure(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.text(0.5, 0.005, NOTICE, ha="center", va="bottom", fontsize=8, color="#8B0000")
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_figures(
    directory: Path,
    metrics: pd.DataFrame,
    data: BatchData,
    contract: AnalysisContract,
    mechanism: pd.DataFrame,
    correlations: pd.DataFrame,
    k_summary: pd.DataFrame,
    drift_summary: pd.DataFrame,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    seeds = metrics[metrics["strategy_id"].isin(data.seed_ids)]
    ensemble = metrics.set_index("strategy_id").loc[str(contract["ensemble_strategy_id"])]
    fields = ["sharpe_ratio", "cagr", "max_drawdown", "mean_monthly_turnover"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.3))
    for ax, field in zip(axes, fields):
        ax.boxplot([seeds[field].to_numpy(float)], positions=[1], showmeans=True)
        ax.scatter([1], [ensemble[field]], color="#D55E00", marker="D", s=38, label="ensemble")
        ax.set_xticks([1], ["20 seeds"])
        ax.set_title(field.replace("_", " "))
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Seed dispersion and arithmetic-ensemble position")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    save_figure(fig, directory, "figure_explanatory_seed_dispersion")

    clean = mechanism.drop(columns=[
        "analysis_classification", "confirmatory_use_permitted", "explanatory_kind"])
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    dates = pd.to_datetime(clean["holding_end_date"])
    for ax, seed_name, ensemble_name, label in [
        (axes[0], "mean_seed_gross_exposure", "ensemble_gross_exposure", "Gross exposure"),
        (axes[1], "mean_seed_short_notional", "ensemble_short_notional", "Short notional"),
        (axes[2], "mean_seed_turnover", "ensemble_turnover", "Turnover"),
    ]:
        ax.plot(dates, clean[seed_name], label="mean individual seed", color="#0072B2")
        ax.plot(dates, clean[ensemble_name], label="arithmetic ensemble", color="#D55E00")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Cross-seed cancellation mechanism")
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    save_figure(fig, directory, "figure_explanatory_ensemble_cancellation")

    corr = correlations.drop(columns=[
        "analysis_classification", "confirmatory_use_permitted", "explanatory_kind",
        "strategy_id"])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(corr.to_numpy(float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title("Pairwise seed weight-path correlations")
    ax.set_xlabel("Seed index")
    ax.set_ylabel("Seed index")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    save_figure(fig, directory, "figure_explanatory_seed_weight_correlations")

    summary = k_summary.drop(columns=[
        "analysis_classification", "confirmatory_use_permitted", "explanatory_kind"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, field, label in [
        (axes[0], "sharpe_ratio", "Sharpe ratio"),
        (axes[1], "mean_gross_exposure", "Mean gross exposure"),
    ]:
        x = summary["k"].to_numpy(int)
        low = summary[f"{field}_bootstrap_q05"].to_numpy(float)
        med = summary[f"{field}_bootstrap_median"].to_numpy(float)
        high = summary[f"{field}_bootstrap_q95"].to_numpy(float)
        deterministic = summary[f"{field}_deterministic_prefix"].to_numpy(float)
        ax.fill_between(x, low, high, color="#56B4E9", alpha=0.3, label="seed-bootstrap 5–95%")
        ax.plot(x, med, color="#0072B2", marker="o", label="seed-bootstrap median")
        ax.plot(x, deterministic, color="#D55E00", marker="s", linestyle="--", label="sorted prefix")
        ax.set_xlabel("Number of seed policies, k")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Exploratory k-seed ensemble sensitivity")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    save_figure(fig, directory, "figure_exploratory_k_seed_sensitivity")

    drift = drift_summary.drop(columns=[
        "analysis_classification", "confirmatory_use_permitted", "explanatory_kind"])
    x = np.arange(len(drift))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(x - width / 2, drift["reported_turnover"], width, label="target-to-target")
    ax.bar(x + width / 2, drift["drift_aware_turnover"], width, label="drift-aware")
    ax.set_xticks(x, drift["group"])
    ax.set_ylabel("Mean monthly full-L1 turnover")
    ax.set_title("Exploratory turnover-convention sensitivity")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    save_figure(fig, directory, "figure_exploratory_drift_turnover_sensitivity")


def output_readme(contract: AnalysisContract) -> str:
    return f"""# Post-holdout explanatory ensemble-mechanism analysis

**{NOTICE}**

These outputs explain a frozen, predeclared arithmetic seed ensemble after
holdout access. They do not amend the locked evaluation, select a new primary
strategy, or support new confirmatory p-values.

- Seed count required: {int(contract['expected_seed_count'])}
- Frozen ensemble: `{contract['ensemble_strategy_id']}`
- Reported scope: `{contract['sample_scope']}`
- k-seed results: deterministic sorted prefixes and a deterministic
  seed-bootstrap with replacement. Both are exploratory because they were
  specified after holdout access.
- Drift-aware turnover: recomputes the previous portfolio after realized asset
  drift while preserving the evaluator's full-L1 convention and economic cost
  rates. It is sensitivity analysis, not a replacement primary result.

All CSV files include `analysis_classification` and
`confirmatory_use_permitted` columns. Every figure carries the same warning.
"""


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def run_analysis(batch_path: Path, contract_path: Path, output_path: Path) -> None:
    contract = AnalysisContract.read(contract_path)
    output_path = output_path.resolve()
    if output_path.exists():
        raise ExplanatoryAnalysisError(
            f"Output already exists and will not be overwritten: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_path.name}_", dir=output_path.parent))
    try:
        data = load_completed_batch(batch_path, contract)
        tables = temporary / "tables"
        figures = temporary / "figures"
        tables.mkdir()
        figures.mkdir()

        metrics = build_metrics(data, contract)
        dispersion, ensemble_comparison, wins = seed_dispersion_tables(
            metrics, data, contract)
        corr_summary, return_corr, weight_corr, diversity = correlation_and_diversity_tables(
            data, contract)
        mechanism_by_period, mechanism_summary = mechanism_tables(
            data, metrics, contract)
        k_draws, k_deterministic, k_summary = k_seed_sensitivity(data, contract)
        drift_details, drift_summary = drift_turnover_sensitivity(data, contract)

        outputs = {
            "explanatory_seed_metrics.csv": labelled(
                metrics, "strategy_metric_recomputation"),
            "explanatory_seed_dispersion.csv": dispersion,
            "explanatory_ensemble_vs_seeds.csv": ensemble_comparison,
            "explanatory_seed_benchmark_win_fractions.csv": wins,
            "explanatory_pairwise_correlation_summary.csv": corr_summary,
            "explanatory_pairwise_return_correlations.csv": return_corr,
            "explanatory_pairwise_weight_correlations.csv": weight_corr,
            "explanatory_weight_diversity.csv": diversity,
            "explanatory_ensemble_mechanism_by_period.csv": mechanism_by_period,
            "explanatory_ensemble_mechanism_summary.csv": mechanism_summary,
            "exploratory_k_seed_bootstrap_draws.csv": k_draws,
            "exploratory_k_seed_deterministic_prefix.csv": k_deterministic,
            "exploratory_k_seed_summary.csv": k_summary,
            "exploratory_drift_turnover_by_strategy.csv": drift_details,
            "exploratory_drift_turnover_summary.csv": drift_summary,
            "input_seed_verification.csv": labelled(
                data.verification, "completed_batch_seed_verification"),
        }
        for name, frame in outputs.items():
            write_csv(frame, tables / name)

        render_figures(
            figures, metrics, data, contract, mechanism_by_period,
            weight_corr, k_summary, drift_summary)
        (temporary / "README.md").write_text(output_readme(contract), encoding="utf-8")
        copied_contract = dict(contract.raw)
        copied_contract["generated_analysis_classification"] = CLASSIFICATION
        copied_contract["confirmatory_use_permitted"] = False
        copied_contract["source_batch"] = str(data.root)
        copied_contract["source_batch_locked_status"] = data.locked_manifest.get("status")
        (temporary / "analysis_contract.json").write_text(
            json.dumps(copied_contract, indent=2, sort_keys=True), encoding="utf-8")
        (temporary / "POST_HOLDOUT_EXPLANATORY_NOTICE.txt").write_text(
            NOTICE + "\nNo output in this directory is a new confirmatory test.\n",
            encoding="utf-8")

        generated = sorted(
            path for path in temporary.rglob("*") if path.is_file())
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "analysis_classification": CLASSIFICATION,
            "confirmatory_use_permitted": False,
            "source_batch": str(data.root),
            "source_locked_batch_manifest_sha256": sha256_file(
                data.root / "locked_batch_manifest.json"),
            "source_scored_panel_sha256": sha256_file(
                data.root / "publication_results" / "raw" / "scored_monthly_panel.csv"),
            "expected_seed_count": int(contract["expected_seed_count"]),
            "verified_seed_count": len(data.seed_ids),
            "ensemble_strategy_id": contract["ensemble_strategy_id"],
            "sample_scope": contract["sample_scope"],
            "generated_files": [
                {"path": str(path.relative_to(temporary)).replace("\\", "/"),
                 "sha256": sha256_file(path)} for path in generated
            ],
        }
        (temporary / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explain a completed locked arithmetic seed ensemble after holdout "
            "access. Outputs are explicitly non-confirmatory."))
    parser.add_argument("--batch", type=Path, required=True,
                        help="Completed locked-batch directory.")
    parser.add_argument("--contract", type=Path, required=True,
                        help="Post-holdout explanatory analysis contract JSON.")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; must not already exist.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(args.batch, args.contract, args.output)
    print(f"{NOTICE}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
