#!/usr/bin/env python3
"""Post-holdout reconciliation of compressed vine-RL ensembles and benchmarks.

This analysis is intentionally labelled exploratory.  The compressed policies were
selected after the locked holdout had been examined, so this script must never be
used to support a fresh confirmatory-superiority claim on that same sample.

The implementation uses only Python's standard library so it can run on a login
node without scientific-Python packages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean, stdev


BENCHMARKS = (
    "equal_weight",
    "shrinkage_mean_variance",
    "dcc_garch",
    "static_vine",
    "rolling_vine",
    "dynamic_nn_vine",
)

VARIANTS = (
    "full_vine_state_and_cvar_observation_ensemble",
    "zero_vine_features_keep_cvar_observation_ensemble",
    "keep_vine_features_zero_cvar_observation_ensemble",
    "zero_vine_features_and_cvar_observation_ensemble",
    "historical_only_no_synthetic_pretraining_ensemble",
)

PRIMARY_EXPLORATORY_VARIANT = (
    "zero_vine_features_keep_cvar_observation_ensemble"
)


class ProtocolError(RuntimeError):
    pass


def truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ProtocolError("Cannot calculate a percentile of an empty sample")
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def crra_utility(monthly_return: float, gamma: float) -> float:
    gross = 1.0 + monthly_return
    if gross <= 0.0:
        raise ProtocolError("CRRA utility is undefined for non-positive gross wealth")
    if abs(gamma - 1.0) < 1e-12:
        return math.log(gross)
    return (gross ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def certainty_equivalent(returns: list[float], gamma: float) -> float:
    utility = mean(crra_utility(value, gamma) for value in returns)
    if abs(gamma - 1.0) < 1e-12:
        return math.exp(utility) - 1.0
    gross = 1.0 + (1.0 - gamma) * utility
    if gross <= 0.0:
        return -1.0
    return gross ** (1.0 / (1.0 - gamma)) - 1.0


def annual_ce(returns: list[float], gamma: float, annual_factor: float) -> float:
    monthly = certainty_equivalent(returns, gamma)
    if monthly <= -1.0:
        return -1.0
    return (1.0 + monthly) ** annual_factor - 1.0


def max_drawdown(returns: list[float]) -> float:
    wealth = 1.0
    peak = 1.0
    worst = 0.0
    for value in returns:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        worst = max(worst, 1.0 - wealth / peak)
    return worst


def metrics(
    returns: list[float],
    turnovers: list[float],
    annual_factor: float,
    gamma: float,
) -> dict[str, float]:
    total = math.prod(1.0 + value for value in returns) - 1.0
    cagr = (1.0 + total) ** (annual_factor / len(returns)) - 1.0
    volatility = stdev(returns) * math.sqrt(annual_factor)
    sharpe = mean(returns) * annual_factor / volatility if volatility > 0 else math.nan
    downside = [min(value, 0.0) for value in returns]
    downside_deviation = math.sqrt(mean(value * value for value in downside))
    sortino = (
        mean(returns) * math.sqrt(annual_factor) / downside_deviation
        if downside_deviation > 0
        else math.inf
    )
    drawdown = max_drawdown(returns)
    calmar = cagr / drawdown if drawdown > 0 else math.inf
    gains = sum(max(value, 0.0) for value in returns)
    losses = -sum(min(value, 0.0) for value in returns)
    omega = gains / losses if losses > 0 else math.inf
    return {
        "observations": float(len(returns)),
        "total_return": total,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": drawdown,
        "calmar_ratio": calmar,
        "omega_ratio": omega,
        "annualized_crra_ce": annual_ce(returns, gamma, annual_factor),
        "mean_monthly_turnover": mean(turnovers),
    }


def circular_indices(n: int, block_length: int, rng: random.Random) -> list[int]:
    indices: list[int] = []
    while len(indices) < n:
        start = rng.randrange(n)
        indices.extend((start + offset) % n for offset in range(block_length))
    return indices[:n]


@dataclass(frozen=True)
class Series:
    strategy_id: str
    experiment_id: str
    dates: tuple[str, ...]
    holding_dates: tuple[str, ...]
    returns: tuple[float, ...]
    turnovers: tuple[float, ...]
    evidence_cohort: str


def load_main(path: Path) -> dict[str, Series]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if truthy(row["is_complete_period"]):
                grouped.setdefault(row["strategy_id"], []).append(row)
    missing = set(BENCHMARKS) - grouped.keys()
    if missing:
        raise ProtocolError(f"Main panel is missing benchmarks: {sorted(missing)}")
    output: dict[str, Series] = {}
    for strategy_id, rows in grouped.items():
        rows.sort(key=lambda item: item["decision_date"])
        output[strategy_id] = Series(
            strategy_id=strategy_id,
            experiment_id=strategy_id,
            dates=tuple(row["decision_date"] for row in rows),
            holding_dates=tuple(row["holding_end_date"] for row in rows),
            returns=tuple(float(row["net_return"]) for row in rows),
            turnovers=tuple(float(row["turnover"]) for row in rows),
            evidence_cohort="locked_main_evaluation",
        )
    return output


def load_causal(path: Path) -> dict[str, Series]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["strategy_level"] == "ensemble" and truthy(row["complete"]):
                grouped.setdefault(row["strategy_id"], []).append(row)
    missing = set(VARIANTS) - grouped.keys()
    if missing:
        raise ProtocolError(f"Causal panel is missing variants: {sorted(missing)}")
    output: dict[str, Series] = {}
    for strategy_id, rows in grouped.items():
        rows.sort(key=lambda item: item["decision_date"])
        output[strategy_id] = Series(
            strategy_id=strategy_id,
            experiment_id=rows[0]["experiment_id"],
            dates=tuple(row["decision_date"] for row in rows),
            holding_dates=tuple(row["holding_end_date"] for row in rows),
            returns=tuple(float(row["net_return"]) for row in rows),
            turnovers=tuple(float(row["turnover"]) for row in rows),
            evidence_cohort="post_holdout_causal_cohort",
        )
    return output


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_inference(
    candidate: Series,
    benchmark: Series,
    gamma: float,
    annual_factor: float,
    block_length: int,
    replications: int,
    seed: int,
) -> dict[str, float | str]:
    if candidate.dates != benchmark.dates or candidate.holding_dates != benchmark.holding_dates:
        raise ProtocolError(
            f"Calendar mismatch between {candidate.strategy_id} and {benchmark.strategy_id}"
        )
    candidate_returns = list(candidate.returns)
    benchmark_returns = list(benchmark.returns)
    utility_differences = [
        crra_utility(left, gamma) - crra_utility(right, gamma)
        for left, right in zip(candidate_returns, benchmark_returns)
    ]
    observed_utility = mean(utility_differences)
    observed_ce = annual_ce(candidate_returns, gamma, annual_factor) - annual_ce(
        benchmark_returns, gamma, annual_factor
    )
    rng = random.Random(seed)
    bootstrap_utility: list[float] = []
    bootstrap_ce: list[float] = []
    centered = [value - observed_utility for value in utility_differences]
    centered_statistics: list[float] = []
    for _ in range(replications):
        indices = circular_indices(len(candidate_returns), block_length, rng)
        candidate_sample = [candidate_returns[index] for index in indices]
        benchmark_sample = [benchmark_returns[index] for index in indices]
        bootstrap_utility.append(
            mean(
                crra_utility(left, gamma) - crra_utility(right, gamma)
                for left, right in zip(candidate_sample, benchmark_sample)
            )
        )
        bootstrap_ce.append(
            annual_ce(candidate_sample, gamma, annual_factor)
            - annual_ce(benchmark_sample, gamma, annual_factor)
        )
        centered_statistics.append(mean(centered[index] for index in indices))
    p_two_sided = (
        1
        + sum(abs(value) >= abs(observed_utility) for value in centered_statistics)
    ) / (replications + 1)
    p_greater = (
        1 + sum(value >= observed_utility for value in centered_statistics)
    ) / (replications + 1)
    return {
        "candidate_strategy_id": candidate.strategy_id,
        "benchmark_strategy_id": benchmark.strategy_id,
        "observations": len(candidate_returns),
        "mean_monthly_crra_utility_difference": observed_utility,
        "annualized_crra_ce_difference": observed_ce,
        "annualized_crra_ce_ci_lower": percentile(bootstrap_ce, 0.025),
        "annualized_crra_ce_ci_upper": percentile(bootstrap_ce, 0.975),
        "two_sided_p_value": p_two_sided,
        "one_sided_candidate_greater_p_value": p_greater,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ProtocolError(f"Refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconcile(main_path: Path, causal_path: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise ProtocolError(f"Output already exists; results are immutable: {output}")
    main = load_main(main_path)
    causal = load_causal(causal_path)
    reference = main[BENCHMARKS[0]]
    for series in list(main.values()) + list(causal.values()):
        if series.dates != reference.dates or series.holding_dates != reference.holding_dates:
            raise ProtocolError(f"Non-identical complete-period calendar: {series.strategy_id}")

    # The complete-period sample excludes two locked calendar months.  Annualize
    # over invested holding intervals, not the wall-clock span containing those
    # omitted periods; this matches the frozen common evaluator.
    holding_days = sum(
        (date.fromisoformat(end) - date.fromisoformat(start)).days
        for start, end in zip(reference.dates, reference.holding_dates)
    )
    elapsed_years = holding_days / 365.0
    annual_factor = len(reference.returns) / elapsed_years
    gamma = 2.0
    block_length = 3
    replications = 9999
    seed = 20261117

    output.mkdir(parents=True)
    economic_rows: list[dict[str, object]] = []
    for series in [main[item] for item in BENCHMARKS] + [causal[item] for item in VARIANTS]:
        row: dict[str, object] = {
            "strategy_id": series.strategy_id,
            "experiment_id": series.experiment_id,
            "evidence_cohort": series.evidence_cohort,
            "selection_status": (
                "frozen_ex_ante"
                if series.evidence_cohort == "locked_main_evaluation"
                else "post_holdout_explanatory"
            ),
        }
        row.update(metrics(list(series.returns), list(series.turnovers), annual_factor, gamma))
        economic_rows.append(row)

    inference_rows = [
        paired_inference(
            causal[PRIMARY_EXPLORATORY_VARIANT],
            main[benchmark],
            gamma,
            annual_factor,
            block_length,
            replications,
            seed + index,
        )
        for index, benchmark in enumerate(BENCHMARKS)
    ]
    adjusted = holm_adjust([float(row["two_sided_p_value"]) for row in inference_rows])
    for row, value in zip(inference_rows, adjusted):
        row["holm_two_sided_p_value"] = value
        lower = float(row["annualized_crra_ce_ci_lower"])
        upper = float(row["annualized_crra_ce_ci_upper"])
        if lower > 0.0 and value <= 0.05:
            conclusion = "exploratory_positive_difference"
        elif upper < 0.0 and value <= 0.05:
            conclusion = "exploratory_negative_difference"
        else:
            conclusion = "difference_not_established"
        row["exploratory_conclusion"] = conclusion
        row["confirmatory_superiority_permitted"] = False

    write_csv(output / "economic_comparison.csv", economic_rows)
    write_csv(output / "paired_crra_inference.csv", inference_rows)
    input_root = output / "input_snapshots"
    input_root.mkdir()
    main_snapshot = input_root / "locked_main_scored_monthly_panel.csv"
    causal_snapshot = input_root / "causal_strategy_period_panel.csv"
    shutil.copy2(main_path, main_snapshot)
    shutil.copy2(causal_path, causal_snapshot)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "analysis_id": "post_hoc_compressed_vine_benchmark_reconciliation_v1",
        "evidence_class": "post_holdout_exploratory",
        "confirmatory_claim_permitted": False,
        "selection_bias_disclosure": (
            "The compressed candidate was selected after inspecting the locked holdout "
            "ablation. Comparisons quantify economic compatibility with frozen benchmarks "
            "but cannot establish fresh out-of-sample superiority."
        ),
        "candidate_interpretation": (
            "The candidate removes raw policy-visible vine features but retains NN-vine "
            "scenario CVaR, CVaR reward shaping, and vine-synthetic pretraining. It is a "
            "compressed vine-plus-RL model, not a no-vine model."
        ),
        "candidate_strategy_id": PRIMARY_EXPLORATORY_VARIANT,
        "benchmark_ids": list(BENCHMARKS),
        "complete_periods": len(reference.returns),
        "elapsed_years": elapsed_years,
        "annualization_factor": annual_factor,
        "crra_gamma": gamma,
        "bootstrap": {
            "method": "paired_circular_moving_block",
            "block_length": block_length,
            "replications": replications,
            "seed": seed,
            "multiplicity": "holm_across_six_post_hoc_benchmark_comparisons",
        },
        "inputs": {
            "locked_main_scored_monthly_panel.csv": {
                "source_path": str(main_path),
                "sha256": sha256(main_snapshot),
                "snapshot": main_snapshot.relative_to(output).as_posix(),
            },
            "causal_strategy_period_panel.csv": {
                "source_path": str(causal_path),
                "sha256": sha256(causal_snapshot),
                "snapshot": causal_snapshot.relative_to(output).as_posix(),
            },
        },
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contents = [
        "economic_comparison.csv", "paired_crra_inference.csv",
        "analysis_manifest.json",
        "input_snapshots/locked_main_scored_monthly_panel.csv",
        "input_snapshots/causal_strategy_period_panel.csv",
    ]
    with (output / "CONTENTS.sha256").open("w", encoding="utf-8", newline="\n") as stream:
        for name in contents:
            stream.write(f"{sha256(output / name)}  {name}\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-panel", required=True, type=Path)
    parser.add_argument("--causal-panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = reconcile(
            arguments.main_panel.resolve(),
            arguments.causal_panel.resolve(),
            arguments.output.resolve(),
        )
    except (OSError, ValueError, ProtocolError) as error:
        print(f"RECONCILIATION FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
