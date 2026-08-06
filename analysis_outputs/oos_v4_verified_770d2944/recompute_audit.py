from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE / "main_oos_v4_operational_retry"
RAW = ROOT / "publication_results" / "raw"
TABLES = ROOT / "publication_results" / "tables"

ASSETS = ["SP500", "NASDAQ", "DOW", "SSE50", "DIVIDEND", "CHINEXT", "GOLD"]
WEIGHT_COLS = [f"w_{asset}" for asset in ASSETS]
GROSS_COLS = [f"g_{asset}" for asset in ASSETS]
MAIN_IDS = [
    "equal_weight",
    "shrinkage_mean_variance",
    "dcc_garch",
    "static_vine",
    "rolling_vine",
    "dynamic_nn_vine",
    "vine_td3_ensemble",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crra_utility(returns: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    gross = 1.0 + returns
    return (np.power(gross, 1.0 - gamma) - 1.0) / (1.0 - gamma)


def monthly_ce(mean_utility: float, gamma: float = 2.0) -> float:
    return np.power(1.0 + (1.0 - gamma) * mean_utility, 1.0 / (1.0 - gamma)) - 1.0


def core_metrics(group: pd.DataFrame) -> dict[str, float]:
    r = group["net_return"].to_numpy(float)
    n = len(r)
    wealth = np.cumprod(1.0 + r)
    total = wealth[-1] - 1.0
    annual_arithmetic = 12.0 * r.mean()
    annual_vol = np.std(r, ddof=1) * np.sqrt(12.0)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])
    drawdown = 1.0 - np.r_[1.0, wealth] / peak
    max_drawdown = drawdown.max()
    cagr = np.power(1.0 + total, 12.0 / n) - 1.0
    utility = crra_utility(r)
    ce_m = monthly_ce(utility.mean())
    return {
        "observations": float(n),
        "total_return": float(total),
        "cagr": float(cagr),
        "annual_arithmetic_return": float(annual_arithmetic),
        "annual_volatility": float(annual_vol),
        "sharpe_ratio": float(annual_arithmetic / annual_vol),
        "max_drawdown": float(max_drawdown),
        "mean_crra_utility": float(utility.mean()),
        "annualized_certainty_equivalent_return": float(np.power(1.0 + ce_m, 12.0) - 1.0),
        "terminal_wealth": float(100000.0 * (1.0 + total)),
        "mean_monthly_turnover": float(group["turnover"].mean()),
        "mean_gross_exposure": float(group["gross_exposure"].mean()),
        "mean_short_notional": float(group["short_notional"].mean()),
    }


panel = pd.read_csv(RAW / "scored_monthly_panel.csv")
panel["is_complete_period"] = panel["is_complete_period"].astype(str).str.lower().eq("true")
complete = panel.loc[panel["is_complete_period"]].copy()
reported = pd.read_csv(TABLES / "table_01_oos_performance.csv").set_index("strategy_id")
metrics_all = pd.read_csv(RAW / "metrics_per_strategy_window_scope.csv")
metrics_complete = metrics_all.loc[metrics_all["sample_scope"].eq("complete_periods")].copy()

recomputed = {sid: core_metrics(g) for sid, g in complete.groupby("strategy_id") if sid in MAIN_IDS}
reconciliation = {}
for sid, values in recomputed.items():
    reconciliation[sid] = {
        key: float(values[key] - reported.loc[sid, key])
        for key in values
        if key in reported.columns
    }

manifest = pd.read_csv(RAW / "validated_strategy_manifest.csv")
seed_manifest = manifest.loc[manifest["role"].eq("proposed")].copy()
seed_ids = seed_manifest["strategy_id"].tolist()
seed_metrics = metrics_complete.loc[metrics_complete["strategy_id"].isin(seed_ids)].set_index("strategy_id")
ensemble = metrics_complete.set_index("strategy_id").loc["vine_td3_ensemble"]
bench = metrics_complete.loc[metrics_complete["strategy_id"].isin(MAIN_IDS[:-1])].set_index("strategy_id")

larger_better = [
    "total_return",
    "cagr",
    "annualized_certainty_equivalent_return",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
]
smaller_better = ["annual_volatility", "max_drawdown", "realized_cvar05_loss", "mean_monthly_turnover"]

seed_outperformance = {}
for benchmark_id, benchmark in bench.iterrows():
    seed_outperformance[benchmark_id] = {}
    for metric in larger_better:
        seed_outperformance[benchmark_id][metric] = float((seed_metrics[metric] > benchmark[metric]).mean())
    for metric in smaller_better:
        seed_outperformance[benchmark_id][metric] = float((seed_metrics[metric] < benchmark[metric]).mean())

ensemble_percentiles = {}
for metric in larger_better + smaller_better + ["mean_gross_exposure", "mean_short_notional"]:
    values = seed_metrics[metric].to_numpy(float)
    ensemble_percentiles[metric] = {
        "ensemble": float(ensemble[metric]),
        "seed_median": float(np.median(values)),
        "seed_min": float(np.min(values)),
        "seed_max": float(np.max(values)),
        "fraction_seeds_below_ensemble": float(np.mean(values < float(ensemble[metric]))),
    }

weight_files = sorted((ROOT / "weights").glob("weights_rl_full_seed_*.csv"))
weight_hashes = {path.name: sha256(path) for path in weight_files}
seed_weights = []
for path in weight_files:
    frame = pd.read_csv(path)
    frame["seed_file"] = path.stem
    seed_weights.append(frame)
weights = pd.concat(seed_weights, ignore_index=True)
weight_mean = weights.groupby(["window_id", "decision_date", "holding_end_date"], as_index=False)[WEIGHT_COLS].mean()
ensemble_panel = panel.loc[panel["strategy_id"].eq("vine_td3_ensemble"), ["window_id", "decision_date", "holding_end_date", *WEIGHT_COLS]]
ensemble_check = weight_mean.merge(
    ensemble_panel,
    on=["window_id", "decision_date", "holding_end_date"],
    suffixes=("_mean_seed", "_reported"),
    validate="one_to_one",
)
ensemble_weight_max_error = max(
    float(np.max(np.abs(ensemble_check[f"{col}_mean_seed"] - ensemble_check[f"{col}_reported"])))
    for col in WEIGHT_COLS
)

per_seed_vectors = {}
for path in weight_files:
    frame = pd.read_csv(path)
    per_seed_vectors[path.stem] = frame[WEIGHT_COLS].to_numpy(float).ravel()
weight_matrix = np.vstack(list(per_seed_vectors.values()))
weight_corr = np.corrcoef(weight_matrix)
off_diag = weight_corr[np.triu_indices_from(weight_corr, k=1)]
pairwise_l1 = []
for i in range(weight_matrix.shape[0]):
    for j in range(i + 1, weight_matrix.shape[0]):
        pairwise_l1.append(np.mean(np.abs(weight_matrix[i] - weight_matrix[j])))

seed_return_panel = complete.loc[complete["strategy_id"].isin(seed_ids)].pivot(
    index="holding_end_date", columns="strategy_id", values="net_return"
)
return_corr = seed_return_panel.corr().to_numpy()
return_off_diag = return_corr[np.triu_indices_from(return_corr, k=1)]

individual_gross = seed_metrics["mean_gross_exposure"].to_numpy(float)
individual_short = seed_metrics["mean_short_notional"].to_numpy(float)
individual_turnover = seed_metrics["mean_monthly_turnover"].to_numpy(float)
cancellation = {
    "mean_individual_seed_gross": float(individual_gross.mean()),
    "ensemble_gross": float(ensemble["mean_gross_exposure"]),
    "gross_reduction": float(individual_gross.mean() - ensemble["mean_gross_exposure"]),
    "mean_individual_seed_short_notional": float(individual_short.mean()),
    "ensemble_short_notional": float(ensemble["mean_short_notional"]),
    "short_reduction": float(individual_short.mean() - ensemble["mean_short_notional"]),
    "mean_individual_seed_turnover": float(individual_turnover.mean()),
    "ensemble_turnover": float(ensemble["mean_monthly_turnover"]),
    "turnover_reduction": float(individual_turnover.mean() - ensemble["mean_monthly_turnover"]),
}

realized = pd.read_csv(ROOT / "inputs" / "realized_asset_gross.csv")
scoring = panel.merge(realized, on=["window_id", "decision_date", "holding_end_date", "trading_days", "is_complete_period"], validate="many_to_one")
gross_recomputed = sum(scoring[w] * (scoring[g] - 1.0) for w, g in zip(WEIGHT_COLS, GROSS_COLS))
gross_return_max_error = float(np.max(np.abs(gross_recomputed - scoring["gross_return"])))
net_recomputed = (1.0 + scoring["gross_return"]) * np.exp(-scoring["transaction_cost"] - scoring["financing_cost"]) - 1.0
net_return_max_error = float(np.max(np.abs(net_recomputed - scoring["net_return"])))

all_constraint = panel.assign(
    net_error=(panel[WEIGHT_COLS].sum(axis=1) - 1.0).abs(),
    gross_from_weights=panel[WEIGHT_COLS].abs().sum(axis=1),
    max_weight=panel[WEIGHT_COLS].max(axis=1),
    min_weight=panel[WEIGHT_COLS].min(axis=1),
)
constraint_summary = {
    "max_net_error": float(all_constraint["net_error"].max()),
    "max_gross": float(all_constraint["gross_from_weights"].max()),
    "max_long": float(all_constraint["max_weight"].max()),
    "min_short": float(all_constraint["min_weight"].min()),
    "violating_rows_at_1e_6": int((
        (all_constraint["net_error"] > 1e-6)
        | (all_constraint["gross_from_weights"] > 1.5 + 1e-6)
        | (all_constraint["max_weight"] > 0.6 + 1e-6)
        | (all_constraint["min_weight"] < -0.2 - 1e-6)
    ).sum()),
}

audit = pd.read_csv(ROOT / "benchmark_weights" / "solver_audit.csv")
opt_audit = audit.loc[audit["convergence"].notna()].copy()
solver_summary = {
    "optimizer_rows": int(len(opt_audit)),
    "convergence_code_counts": {str(int(k)): int(v) for k, v in opt_audit["convergence"].value_counts().sort_index().items()},
    "maxeval_code_5_by_method": {
        method: int((group["convergence"] == 5).sum()) for method, group in opt_audit.groupby("method")
    },
    "future_input_rows": int((pd.to_datetime(audit["latest_input_date"]) > pd.to_datetime(audit["decision_date"])).sum()),
    "missing_objective_optimizer_rows": int(opt_audit["objective"].isna().sum()),
}

implementation = pd.read_csv(TABLES / "table_04_economic_implementation.csv").set_index("strategy_id")
implementation_effect = {}
for sid, row in implementation.iterrows():
    gross_before = float(row["gross_total_return_before_costs"])
    drag = float(row["implementation_drag_total_return"])
    implementation_effect[sid] = {
        "drag_total_return_points": drag,
        "drag_fraction_of_pre_cost_total_return": float(drag / gross_before) if gross_before else 0.0,
    }

main_complete = metrics_complete.loc[metrics_complete["strategy_id"].isin(MAIN_IDS)].set_index("strategy_id")
main_ranks = {}
for metric in larger_better:
    main_ranks[metric] = int(main_complete[metric].rank(ascending=False, method="min").loc["vine_td3_ensemble"])
for metric in smaller_better:
    main_ranks[metric] = int(main_complete[metric].rank(ascending=True, method="min").loc["vine_td3_ensemble"])

seed_extremes = {}
for metric in larger_better + smaller_better + ["mean_gross_exposure", "mean_short_notional"]:
    series = seed_metrics[metric].astype(float)
    seed_extremes[metric] = {
        "minimum_seed": str(series.idxmin()),
        "minimum": float(series.min()),
        "maximum_seed": str(series.idxmax()),
        "maximum": float(series.max()),
    }

candidate_returns = complete.loc[complete["strategy_id"].eq("vine_td3_ensemble"), ["holding_end_date", "net_return"]].set_index("holding_end_date")["net_return"]
active_return_diagnostics = {}
for benchmark_id in MAIN_IDS[:-1]:
    benchmark_returns = complete.loc[complete["strategy_id"].eq(benchmark_id), ["holding_end_date", "net_return"]].set_index("holding_end_date")["net_return"]
    aligned = pd.concat([candidate_returns.rename("candidate"), benchmark_returns.rename("benchmark")], axis=1).dropna()
    active = aligned["candidate"] - aligned["benchmark"]
    leave_one_out_differences = []
    for date in aligned.index:
        reduced = aligned.drop(index=date)
        candidate_total = float(np.prod(1.0 + reduced["candidate"]) - 1.0)
        benchmark_total = float(np.prod(1.0 + reduced["benchmark"]) - 1.0)
        leave_one_out_differences.append(candidate_total - benchmark_total)
    top = active.sort_values(ascending=False)
    bottom = active.sort_values(ascending=True)
    active_return_diagnostics[benchmark_id] = {
        "mean_monthly_active_return": float(active.mean()),
        "active_return_positive_month_fraction": float((active > 0).mean()),
        "largest_positive_active_months": {str(k): float(v) for k, v in top.iloc[:3].items()},
        "largest_negative_active_months": {str(k): float(v) for k, v in bottom.iloc[:3].items()},
        "leave_one_month_out_total_return_difference_min": float(np.min(leave_one_out_differences)),
        "leave_one_month_out_total_return_difference_max": float(np.max(leave_one_out_differences)),
        "leave_one_month_out_fraction_candidate_ahead": float(np.mean(np.asarray(leave_one_out_differences) > 0.0)),
    }


def drift_aware_rescore(strategy_group: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc accounting sensitivity; it does not replace the frozen result."""
    rows = []
    previous_pretrade = np.repeat(1.0 / len(ASSETS), len(ASSETS))
    for _, row in strategy_group.sort_values("decision_date", kind="stable").iterrows():
        w = row[WEIGHT_COLS].to_numpy(float)
        g = row[GROSS_COLS].to_numpy(float)
        turnover = float(np.abs(w - previous_pretrade).sum())
        transaction_cost = 0.001 * turnover
        short_notional = float(np.maximum(-w, 0.0).sum())
        financing_cost = (0.03 * short_notional + 0.02 * max(float(w.sum()) - 1.0, 0.0)) / 12.0
        gross_portfolio = 1.0 + float(np.dot(w, g - 1.0))
        net_return = gross_portfolio * np.exp(-transaction_cost - financing_cost) - 1.0
        rows.append({
            "decision_date": row["decision_date"],
            "holding_end_date": row["holding_end_date"],
            "is_complete_period": bool(row["is_complete_period"]),
            "net_return": net_return,
            "turnover": turnover,
            "gross_exposure": float(np.abs(w).sum()),
            "short_notional": short_notional,
        })
        previous_pretrade = w * g / gross_portfolio
    return pd.DataFrame(rows)


panel_with_realized = panel.merge(
    realized,
    on=["window_id", "decision_date", "holding_end_date", "trading_days", "is_complete_period"],
    validate="many_to_one",
)
drift_aware_sensitivity = {}
for strategy_id in MAIN_IDS:
    rescored_all = drift_aware_rescore(panel_with_realized.loc[panel_with_realized["strategy_id"].eq(strategy_id)])
    rescored_complete = rescored_all.loc[rescored_all["is_complete_period"]]
    corrected = core_metrics(rescored_complete)
    frozen = reported.loc[strategy_id]
    drift_aware_sensitivity[strategy_id] = {
        "corrected_total_return": corrected["total_return"],
        "frozen_total_return": float(frozen["total_return"]),
        "total_return_change": corrected["total_return"] - float(frozen["total_return"]),
        "corrected_cagr": corrected["cagr"],
        "corrected_sharpe": corrected["sharpe_ratio"],
        "corrected_annualized_ce": corrected["annualized_certainty_equivalent_return"],
        "corrected_mean_monthly_turnover": corrected["mean_monthly_turnover"],
    }

output = {
    "reconciliation_max_abs_error": {
        key: float(max(abs(v.get(key, 0.0)) for v in reconciliation.values()))
        for key in sorted({k for values in reconciliation.values() for k in values})
    },
    "gross_return_formula_max_error": gross_return_max_error,
    "net_return_formula_max_error": net_return_max_error,
    "constraint_summary_all_648_strategy_period_rows": constraint_summary,
    "seed_count": int(len(seed_ids)),
    "unique_seed_weight_hashes": int(len(set(weight_hashes.values()))),
    "unique_checkpoint_hashes": int(seed_manifest["checkpoint_sha256"].nunique()),
    "ensemble_weight_max_error_vs_arithmetic_mean": ensemble_weight_max_error,
    "seed_weight_pairwise_correlation": {
        "min": float(off_diag.min()), "median": float(np.median(off_diag)), "max": float(off_diag.max())
    },
    "seed_weight_pairwise_mean_absolute_difference": {
        "min": float(np.min(pairwise_l1)), "median": float(np.median(pairwise_l1)), "max": float(np.max(pairwise_l1))
    },
    "seed_return_pairwise_correlation": {
        "min": float(return_off_diag.min()), "median": float(np.median(return_off_diag)), "max": float(return_off_diag.max())
    },
    "ensemble_cancellation": cancellation,
    "ensemble_relative_to_seed_distribution": ensemble_percentiles,
    "fraction_of_seeds_outperforming_each_benchmark": seed_outperformance,
    "solver_summary": solver_summary,
    "implementation_effect": implementation_effect,
    "ensemble_rank_among_seven_main_strategies": main_ranks,
    "seed_extremes": seed_extremes,
    "active_return_diagnostics": active_return_diagnostics,
    "posthoc_drift_aware_turnover_sensitivity": drift_aware_sensitivity,
}

print(json.dumps(output, indent=2, sort_keys=True))
