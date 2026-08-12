#!/usr/bin/env python3
"""Analyze the standardized, common-accounting causal evaluation panel.

This script never generates returns or silently drops strategies. It requires
all 130 seed policies and all 13 weight-space ensembles on identical periods.
Training seeds quantify optimization stability, not market-sample uncertainty.
"""

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

from publication_pipeline_draft.causal_analysis_contract import (
    CausalAnalysisContractError,
    load_contract,
    require,
)


class CausalAnalysisError(CausalAnalysisContractError):
    pass


NUMERIC_COLUMNS = (
    "gross_return", "net_return", "turnover", "transaction_cost",
    "financing_cost", "gross_exposure", "net_exposure", "short_notional",
    "max_abs_weight", "gross_constraint_violation", "net_constraint_violation",
    "position_constraint_violation",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(series: pd.Series, label: str) -> pd.Series:
    mapping = {"true": True, "1": True, "yes": True,
               "false": False, "0": False, "no": False}
    values = series.astype(str).str.strip().str.lower().map(mapping)
    require(values.notna().all(), f"{label} contains invalid Boolean values.")
    return values.astype(bool)


def read_period_panel(path: Path, contract: dict[str, Any]) -> pd.DataFrame:
    require(path.is_file(), f"Causal period panel not found: {path}")
    frame = pd.read_csv(path)
    required = set(contract["required_period_columns"])
    require(required <= set(frame.columns),
            f"Causal period panel lacks columns: {sorted(required - set(frame.columns))}")
    frame = frame[list(contract["required_period_columns"])].copy()
    for name in ("decision_date", "holding_end_date"):
        frame[name] = pd.to_datetime(frame[name], errors="raise")
    for name in NUMERIC_COLUMNS:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    require(np.isfinite(frame[list(NUMERIC_COLUMNS)].to_numpy(float)).all(),
            "Causal period panel contains non-finite numeric values.")
    frame["complete"] = parse_bool(frame["complete"], "complete")
    require(frame["complete"].all(), "Incomplete periods are forbidden by this contract.")
    require((frame["holding_end_date"] > frame["decision_date"]).all(),
            "Holding periods must be positive.")
    require((frame["net_return"] > -1).all() and (frame["gross_return"] > -1).all(),
            "Returns below -100 percent are invalid.")
    require(set(frame["window_id"].astype(str)) == {contract["sample"]["window_id"]},
            "Period panel has an unexpected window identifier.")
    require(set(frame["strategy_level"].astype(str)) == {"seed", "ensemble"},
            "strategy_level must contain seed and ensemble only.")
    return frame


def validate_panel(frame: pd.DataFrame, contract: dict[str, Any]) -> None:
    experiments = {contract["reference_experiment_id"]}
    experiments |= {item["alternative_experiment_id"]
                    for item in contract["primary_component_contrasts"]}
    experiments |= {item["alternative_experiment_id"]
                    for item in contract["algorithm_robustness_contrasts"]}
    require(set(frame["experiment_id"].astype(str)) == experiments,
            "Period panel does not contain exactly thirteen experiments.")
    expected_seeds = set(int(value) for value in contract["expected_seeds"])
    expected_periods = int(contract["sample"]["expected_periods"])
    tolerance = float(contract["economics"]["weight_tolerance"])

    canonical_dates: pd.DataFrame | None = None
    strategy_keys: set[tuple[str, str, int | None]] = set()
    for experiment in sorted(experiments):
        subset = frame[frame["experiment_id"] == experiment]
        seed_rows = subset[subset["strategy_level"] == "seed"].copy()
        seed_rows["seed"] = pd.to_numeric(seed_rows["seed"], errors="raise").astype(int)
        require(set(seed_rows["seed"]) == expected_seeds,
                f"{experiment} does not contain the exact matched seeds.")
        for seed, group in seed_rows.groupby("seed", sort=True):
            require(group["strategy_id"].nunique() == 1 and len(group) == expected_periods,
                    f"{experiment} seed {seed} has invalid strategy/period cardinality.")
            strategy_keys.add((experiment, "seed", int(seed)))
        ensemble = subset[subset["strategy_level"] == "ensemble"]
        require(ensemble["strategy_id"].nunique() == 1 and len(ensemble) == expected_periods,
                f"{experiment} must contain exactly one complete ensemble path.")
        require(ensemble["seed"].isna().all() |
                ensemble["seed"].astype(str).str.strip().isin({"", "nan", "NA"}).all(),
                f"{experiment} ensemble must not be assigned a training seed.")
        strategy_keys.add((experiment, "ensemble", None))
    require(len(strategy_keys) == 143, "Expected 130 seed paths plus 13 ensembles.")

    for _, group in frame.groupby("strategy_id", sort=True):
        group = group.sort_values(["decision_date", "holding_end_date"])
        require(len(group) == expected_periods, "Every strategy must have 24 periods.")
        dates = group[["decision_date", "holding_end_date"]].reset_index(drop=True)
        if canonical_dates is None:
            canonical_dates = dates
        else:
            require(dates.equals(canonical_dates), "Strategies do not share identical dates.")
    require(frame["strategy_id"].nunique() == 143,
            "strategy_id must uniquely identify all 143 paths.")
    require(frame["gross_constraint_violation"].abs().max() <= tolerance,
            "Gross-exposure constraint violation detected.")
    require(frame["net_constraint_violation"].abs().max() <= tolerance,
            "Net-exposure constraint violation detected.")
    require(frame["position_constraint_violation"].abs().max() <= tolerance,
            "Position constraint violation detected.")
    economics = contract["economics"]
    require(frame["gross_exposure"].max() <= float(economics["gross_leverage"]) + tolerance,
            "Gross exposure exceeds the contract.")
    require((frame["net_exposure"] - float(economics["net_exposure"])).abs().max() <= tolerance,
            "Net exposure differs from the contract.")
    require(frame["max_abs_weight"].max() <=
            max(float(economics["max_long_weight"]),
                float(economics["max_short_weight"])) + tolerance,
            "Position magnitude exceeds the contract.")


def annualization(group: pd.DataFrame, contract: dict[str, Any]) -> tuple[float, float]:
    years = ((group["holding_end_date"] - group["decision_date"]).dt.days / 365.0)
    require((years > 0).all(), "Annualization contains non-positive holding periods.")
    elapsed = float(years.sum())
    return float(len(group) / elapsed), elapsed


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


def metrics(group: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    group = group.sort_values("decision_date")
    returns = group["net_return"].to_numpy(float)
    gross_returns = group["gross_return"].to_numpy(float)
    factor, elapsed = annualization(group, contract)
    wealth = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])
    drawdown = np.r_[1.0, wealth] / peaks - 1.0
    annual_rf = float(contract["economics"]["annual_risk_free_rate"])
    period_rf = (1.0 + annual_rf) ** (1.0 / factor) - 1.0
    excess = returns - period_rf
    volatility = float(np.std(returns, ddof=1))
    downside = np.minimum(excess, 0.0)
    downside_deviation = math.sqrt(float(np.mean(downside ** 2)))
    loss_quantile = float(np.quantile(returns, 0.05))
    tail = returns[returns <= loss_quantile]
    omega_denominator = float(-np.minimum(excess, 0.0).sum())
    gamma = float(contract["economics"]["crra_gamma"])
    result = {
        "observations": len(group), "elapsed_years": elapsed,
        "total_return": float(wealth[-1] - 1.0),
        "cagr": float(wealth[-1] ** (1.0 / elapsed) - 1.0),
        "annual_volatility": volatility * math.sqrt(factor),
        "sharpe_ratio": (float(excess.mean()) / float(np.std(excess, ddof=1)) *
                         math.sqrt(factor)) if np.std(excess, ddof=1) > 0 else math.nan,
        "sortino_ratio": (float(excess.mean()) / downside_deviation * math.sqrt(factor))
                         if downside_deviation > 0 else math.nan,
        "max_drawdown": float(-drawdown.min()),
        "calmar_ratio": (float(wealth[-1] ** (1.0 / elapsed) - 1.0) /
                         float(-drawdown.min())) if drawdown.min() < 0 else math.nan,
        "omega_ratio": (float(np.maximum(excess, 0.0).sum()) / omega_denominator)
                       if omega_denominator > 0 else math.nan,
        "monthly_var_95_loss": -loss_quantile,
        "monthly_cvar_95_loss": -float(tail.mean()),
        "annualized_certainty_equivalent_return": annualized_ce(returns, gamma, factor),
        "mean_crra_utility": float(crra_utility(returns, gamma).mean()),
        "terminal_wealth": float(contract["economics"]["initial_wealth"]) * wealth[-1],
        "gross_total_return_before_costs": float(np.prod(1.0 + gross_returns) - 1.0),
        "mean_monthly_turnover": float(group["turnover"].mean()),
        "annualized_turnover": float(group["turnover"].mean() * factor),
        "mean_gross_exposure": float(group["gross_exposure"].mean()),
        "maximum_gross_exposure": float(group["gross_exposure"].max()),
        "mean_short_notional": float(group["short_notional"].mean()),
        "total_transaction_cost": float(group["transaction_cost"].sum()),
        "total_financing_cost": float(group["financing_cost"].sum()),
        "implementation_drag_total_return": float(
            np.prod(1.0 + gross_returns) - np.prod(1.0 + returns)),
    }
    return result


def circular_indices(rng: np.random.Generator, n: int, block: int) -> np.ndarray:
    starts = rng.integers(0, n, size=math.ceil(n / block))
    return np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def ensemble_group(frame: pd.DataFrame, experiment: str) -> pd.DataFrame:
    group = frame[(frame["experiment_id"] == experiment) &
                  (frame["strategy_level"] == "ensemble")]
    require(bool(len(group)), f"Missing ensemble: {experiment}")
    return group.sort_values("decision_date")


def contrast_rows(frame: pd.DataFrame, contract: dict[str, Any],
                  items: list[dict[str, str]], family: str,
                  two_sided: bool) -> pd.DataFrame:
    reference_id = contract["reference_experiment_id"]
    reference = ensemble_group(frame, reference_id)
    ref_returns = reference["net_return"].to_numpy(float)
    factor, _ = annualization(reference, contract)
    gamma = float(contract["economics"]["crra_gamma"])
    settings = contract["inference"]
    replications = int(settings["bootstrap_replications"])
    block = int(settings["bootstrap_block_length"])
    seed = int(settings["inference_seed"])
    confidence = float(settings["confidence_level"])
    rows: list[dict[str, Any]] = []
    for number, item in enumerate(items):
        alternative_id = item["alternative_experiment_id"]
        alternative = ensemble_group(frame, alternative_id)
        alt_returns = alternative["net_return"].to_numpy(float)
        difference = crra_utility(ref_returns, gamma) - crra_utility(alt_returns, gamma)
        observed_utility = float(difference.mean())
        observed_ce = annualized_ce(ref_returns, gamma, factor) - annualized_ce(
            alt_returns, gamma, factor)
        rng = np.random.default_rng(seed + 10007 * (number + 1) +
                                    (500000 if two_sided else 0))
        centered = difference - observed_utility
        null_statistics = np.empty(replications)
        ce_statistics = np.empty(replications)
        for replication in range(replications):
            sample = circular_indices(rng, len(difference), block)
            null_statistics[replication] = centered[sample].mean()
            ce_statistics[replication] = (
                annualized_ce(ref_returns[sample], gamma, factor) -
                annualized_ce(alt_returns[sample], gamma, factor))
        if two_sided:
            p_value = (
                1 + int(np.sum(np.abs(null_statistics) >= abs(observed_utility)))
            ) / (replications + 1)
        else:
            p_value = (
                1 + int(np.sum(null_statistics >= observed_utility))
            ) / (replications + 1)
        alpha_tail = (1.0 - confidence) / 2.0
        rows.append({
            "contrast_family": family, "label": item["label"],
            "reference_experiment_id": reference_id,
            "alternative_experiment_id": alternative_id,
            "observed_mean_crra_utility_difference": observed_utility,
            "annualized_ce_difference": observed_ce,
            "annualized_ce_ci_lower": float(np.quantile(ce_statistics, alpha_tail)),
            "annualized_ce_ci_upper": float(np.quantile(ce_statistics, 1.0 - alpha_tail)),
            "raw_p_value": p_value,
            "reference_total_return": float(np.prod(1.0 + ref_returns) - 1.0),
            "alternative_total_return": float(np.prod(1.0 + alt_returns) - 1.0),
            "reference_mean_turnover": float(reference["turnover"].mean()),
            "alternative_mean_turnover": float(alternative["turnover"].mean()),
            "test_sidedness": "two_sided" if two_sided else "one_sided_reference_greater",
            "bootstrap_replications": replications, "bootstrap_block_length": block,
        })
    adjusted = holm_adjust([float(row["raw_p_value"]) for row in rows])
    alpha = 1.0 - confidence
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p_value"] = value
        row["reject_after_holm_5pct"] = bool(value <= alpha)
        if two_sided:
            if value <= alpha and row["annualized_ce_difference"] > 0:
                row["contract_decision"] = "exploratory_reference_better"
            elif value <= alpha and row["annualized_ce_difference"] < 0:
                row["contract_decision"] = "exploratory_alternative_better"
            else:
                row["contract_decision"] = "exploratory_difference_not_established"
        elif value <= alpha and row["annualized_ce_difference"] > 0:
            row["contract_decision"] = "component_supported"
        elif row["annualized_ce_ci_upper"] < 0:
            row["contract_decision"] = "opposite_direction_evidence"
        else:
            row["contract_decision"] = "component_not_established"
    return pd.DataFrame(rows)


def seed_effects(frame: pd.DataFrame, contract: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_id = contract["reference_experiment_id"]
    gamma = float(contract["economics"]["crra_gamma"])
    all_items = contract["primary_component_contrasts"] + \
        contract["algorithm_robustness_contrasts"]
    raw: list[dict[str, Any]] = []
    for item in all_items:
        alternative_id = item["alternative_experiment_id"]
        for seed in contract["expected_seeds"]:
            reference = frame[(frame["experiment_id"] == reference_id) &
                              (frame["strategy_level"] == "seed") &
                              (pd.to_numeric(frame["seed"], errors="coerce") == seed)]
            alternative = frame[(frame["experiment_id"] == alternative_id) &
                                (frame["strategy_level"] == "seed") &
                                (pd.to_numeric(frame["seed"], errors="coerce") == seed)]
            factor, _ = annualization(reference, contract)
            effect = annualized_ce(reference["net_return"].to_numpy(float), gamma, factor) - \
                annualized_ce(alternative["net_return"].to_numpy(float), gamma, factor)
            raw.append({"label": item["label"],
                        "alternative_experiment_id": alternative_id,
                        "seed": seed, "paired_annualized_ce_difference": effect,
                        "inference_scope": "training_randomness_only"})
    raw_frame = pd.DataFrame(raw)
    summaries = []
    for (label, alternative_id), group in raw_frame.groupby(
            ["label", "alternative_experiment_id"], sort=False):
        values = group["paired_annualized_ce_difference"].to_numpy(float)
        summaries.append({"label": label, "alternative_experiment_id": alternative_id,
                          "matched_seed_count": len(values), "mean": values.mean(),
                          "median": np.median(values), "standard_deviation": values.std(ddof=1),
                          "minimum": values.min(), "maximum": values.max(),
                          "fraction_reference_positive": np.mean(values > 0),
                          "inference_scope": "training_randomness_only_not_market_uncertainty"})
    return raw_frame, pd.DataFrame(summaries)


def write_plots(output: Path, frame: pd.DataFrame, metrics_frame: pd.DataFrame,
                primary: pd.DataFrame, seed_pairs: pd.DataFrame,
                contract: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 300,
                         "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})

    def save(fig: Any, stem: str) -> None:
        fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight",
                    facecolor="white")
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)

    ordered = primary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    values = ordered["annualized_ce_difference"].to_numpy(float)
    positions = np.arange(len(ordered))
    ax.hlines(positions,
              ordered["annualized_ce_ci_lower"].to_numpy(float),
              ordered["annualized_ce_ci_upper"].to_numpy(float),
              color="#6C8EBF", linewidth=1.5)
    ax.scatter(values, positions, color="#17365D", s=25, zorder=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["label"])
    ax.set_xlabel("Reference minus ablation: annualized CRRA certainty equivalent")
    ax.set_title("Preregistered causal component effects (post-holdout explanatory)")
    fig.tight_layout(); save(fig, "causal_crra_effect_forest")

    primary_ids = {item["alternative_experiment_id"]
                   for item in contract["primary_component_contrasts"]}
    seed_primary = seed_pairs[seed_pairs["alternative_experiment_id"].isin(primary_ids)]
    labels = [item["label"] for item in contract["primary_component_contrasts"]]
    values = [seed_primary[seed_primary["label"] == label][
        "paired_annualized_ce_difference"].to_numpy(float) for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    # ``labels`` remains compatible with the older Matplotlib used on the HPC;
    # newer releases merely deprecate it in favour of ``tick_labels``.
    ax.boxplot(values, labels=labels, showmeans=True)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Matched-seed annualized CE difference")
    ax.set_title("Optimization-seed stability (not market-sample inference)")
    fig.tight_layout(); save(fig, "causal_seed_stability")

    component_ids = [contract["reference_experiment_id"]] + [
        item["alternative_experiment_id"] for item in contract["primary_component_contrasts"]]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for experiment in component_ids:
        group = ensemble_group(frame, experiment)
        wealth = np.cumprod(1.0 + group["net_return"].to_numpy(float))
        ax.plot(group["holding_end_date"], wealth, label=experiment, linewidth=1.2)
    ax.set_ylabel("Net wealth multiple")
    ax.set_title("Causal component ensemble wealth paths")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); save(fig, "causal_ensemble_wealth")

    ensemble_metrics = metrics_frame[metrics_frame["strategy_level"] == "ensemble"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(ensemble_metrics["annualized_turnover"],
               ensemble_metrics["annualized_certainty_equivalent_return"],
               color="#2F5597")
    for _, row in ensemble_metrics.iterrows():
        ax.annotate(row["experiment_id"],
                    (row["annualized_turnover"],
                     row["annualized_certainty_equivalent_return"]), fontsize=5)
    ax.set_xlabel("Annualized turnover")
    ax.set_ylabel("Annualized CRRA certainty equivalent")
    ax.set_title("Economic performance versus implementation intensity")
    fig.tight_layout(); save(fig, "causal_turnover_performance")


def analyze(contract_path: Path, period_panel: Path, output: Path) -> dict[str, Any]:
    validated = load_contract(contract_path)
    require(not output.exists(), f"Analysis output already exists: {output}")
    frame = read_period_panel(period_panel, validated.raw)
    validate_panel(frame, validated.raw)

    metric_rows: list[dict[str, Any]] = []
    group_columns = ["experiment_id", "strategy_id", "strategy_level"]
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        seed_values = pd.to_numeric(group["seed"], errors="coerce").dropna().unique()
        metric_rows.append({"experiment_id": key[0], "strategy_id": key[1],
                            "strategy_level": key[2],
                            "seed": int(seed_values[0]) if len(seed_values) else None,
                            **metrics(group, validated.raw)})
    metric_frame = pd.DataFrame(metric_rows)
    primary = contrast_rows(frame, validated.raw,
                            validated.raw["primary_component_contrasts"],
                            "primary_component", two_sided=False)
    algorithms = contrast_rows(frame, validated.raw,
                               validated.raw["algorithm_robustness_contrasts"],
                               "algorithm_robustness", two_sided=True)
    seed_pairs, seed_summary = seed_effects(frame, validated.raw)
    ensemble_metrics = metric_frame[metric_frame["strategy_level"] == "ensemble"].copy()
    seed_means = metric_frame[metric_frame["strategy_level"] == "seed"].groupby(
        "experiment_id", as_index=False).agg(
            seed_mean_turnover=("mean_monthly_turnover", "mean"),
            seed_sd_turnover=("mean_monthly_turnover", "std"),
            seed_mean_gross_exposure=("mean_gross_exposure", "mean"),
            seed_mean_short_notional=("mean_short_notional", "mean"),
            seed_mean_transaction_cost=("total_transaction_cost", "mean"),
            seed_mean_financing_cost=("total_financing_cost", "mean"))
    implementation = ensemble_metrics[[
        "experiment_id", "mean_monthly_turnover", "annualized_turnover",
        "mean_gross_exposure", "maximum_gross_exposure", "mean_short_notional",
        "total_transaction_cost", "total_financing_cost"]].merge(
            seed_means, on="experiment_id", validate="one_to_one")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        tables = {
            "causal_strategy_metrics.csv": metric_frame,
            "causal_primary_contrasts.csv": primary,
            "causal_algorithm_robustness.csv": algorithms,
            "causal_seed_stability.csv": seed_summary,
            "causal_seed_pair_effects.csv": seed_pairs,
            "causal_implementation_diagnostics.csv": implementation,
        }
        for name, table in tables.items():
            table.to_csv(temporary / name, index=False)
        write_plots(temporary, frame, metric_frame, primary, seed_pairs, validated.raw)
        manifest = {
            "schema_version": 1, "status": "causal_analysis_complete",
            "analysis_id": validated.raw["analysis_id"],
            "analysis_contract_sha256": validated.sha256,
            "period_panel_sha256": sha256(period_panel),
            "period_rows": len(frame), "strategy_count": frame["strategy_id"].nunique(),
            "experiment_count": frame["experiment_id"].nunique(),
            "primary_contrast_count": len(primary),
            "algorithm_contrast_count": len(algorithms),
            "bootstrap_replications_per_contrast": validated.raw["inference"][
                "bootstrap_replications"],
            "all_preregistered_results_reported": True,
            "seed_inference_scope": validated.raw["sample"]["seed_inference_scope"],
            "evidence_class": validated.raw["evidence_class"],
            "confirmatory_claim_permitted": False,
        }
        (temporary / "causal_analysis_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.copy2(contract_path, temporary / "causal_analysis_contract_v1.json")
        required_names = set(validated.raw["required_outputs"]["tables"] +
                             validated.raw["required_outputs"]["figures"])
        require(required_names <= {path.name for path in temporary.iterdir()},
                "A preregistered analysis output was not generated.")
        checksum_lines = []
        for path in sorted(temporary.iterdir()):
            if path.is_file() and path.name != "CONTENTS.sha256":
                checksum_lines.append(f"{sha256(path)}  {path.name}")
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="ascii")
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v1.json"))
    parser.add_argument("--period-panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.contract.resolve(), args.period_panel.resolve(), args.output)
    except (CausalAnalysisContractError, OSError, ValueError) as error:
        print(f"CAUSAL ANALYSIS FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
