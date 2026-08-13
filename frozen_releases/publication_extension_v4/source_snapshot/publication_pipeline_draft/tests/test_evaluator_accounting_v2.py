from __future__ import annotations

import math

import numpy as np
import pandas as pd

from publication_pipeline_draft.publication_pipeline import (
    Contract,
    empirical_metrics,
    score_strategy,
)


def base_contract(**updates: object) -> Contract:
    values: dict[str, object] = {
        "periods_per_year": 12,
        "initial_wealth": 100000.0,
        "net_exposure": 1.0,
        "turnover_cost": 0.001,
        "annual_short_borrow_rate": 0.03,
        "annual_cash_borrow_rate": 0.02,
    }
    values.update(updates)
    return Contract(values)


def panels(weights: list[list[float]], grosses: list[list[float]], days: list[int]):
    dates = pd.to_datetime(["2030-01-31", "2030-02-28"])
    realized = pd.DataFrame(
        {
            "window_id": ["future_test", "future_test"],
            "decision_date": dates,
            "holding_end_date": dates + pd.to_timedelta(days, unit="D"),
            "trading_days": days,
            "is_complete_period": [True, True],
            "g_A": [item[0] for item in grosses],
            "g_B": [item[1] for item in grosses],
        }
    )
    weight_frame = realized[["window_id", "decision_date", "holding_end_date"]].copy()
    weight_frame["w_A"] = [item[0] for item in weights]
    weight_frame["w_B"] = [item[1] for item in weights]
    return weight_frame, realized


def test_drifted_turnover_uses_realized_pretrade_holdings() -> None:
    weights, realized = panels(
        [[0.8, 0.2], [0.8, 0.2]],
        [[2.0, 1.0], [1.0, 1.0]],
        [21, 21],
    )
    legacy = score_strategy(
        "legacy", weights, realized, ["A", "B"], base_contract()
    )
    drifted = score_strategy(
        "drifted",
        weights,
        realized,
        ["A", "B"],
        base_contract(turnover_convention="drifted_pretrade_v1"),
    )
    assert math.isclose(float(legacy.iloc[1]["turnover"]), 0.0, abs_tol=1e-15)
    expected_pretrade = np.array([1.6, 0.2]) / 1.8
    expected = float(np.abs(np.array([0.8, 0.2]) - expected_pretrade).sum())
    assert math.isclose(float(drifted.iloc[1]["turnover"]), expected, rel_tol=1e-12)


def test_partial_period_financing_is_day_prorated() -> None:
    weights, realized = panels(
        [[1.2, -0.2], [1.2, -0.2]],
        [[1.0, 1.0], [1.0, 1.0]],
        [21, 3],
    )
    scored = score_strategy(
        "prorated",
        weights,
        realized,
        ["A", "B"],
        base_contract(financing_proration="trading_days_v1", annual_trading_days=252),
    )
    annual_charge = 0.03 * 0.2
    assert math.isclose(
        float(scored.iloc[0]["financing_cost"]), annual_charge * 21 / 252, rel_tol=1e-12
    )
    assert math.isclose(
        float(scored.iloc[1]["financing_cost"]), annual_charge * 3 / 252, rel_tol=1e-12
    )
    assert float(scored.iloc[1]["financing_cost"]) < float(scored.iloc[0]["financing_cost"])


def test_financing_can_use_actual_calendar_days() -> None:
    weights, realized = panels(
        [[1.2, -0.2], [1.2, -0.2]],
        [[1.0, 1.0], [1.0, 1.0]],
        [31, 3],
    )
    scored = score_strategy(
        "actual_days",
        weights,
        realized,
        ["A", "B"],
        base_contract(financing_proration="actual_calendar_days_v1", day_count_basis=365),
    )
    annual_charge = 0.03 * 0.2
    assert math.isclose(
        float(scored.iloc[0]["financing_cost"]), annual_charge * 31 / 365, rel_tol=1e-12
    )
    assert math.isclose(
        float(scored.iloc[1]["financing_cost"]), annual_charge * 3 / 365, rel_tol=1e-12
    )


def test_future_annualization_uses_actual_elapsed_calendar_time() -> None:
    weights, realized = panels(
        [[0.6, 0.4], [0.6, 0.4]],
        [[1.02, 1.02], [1.01, 1.01]],
        [31, 3],
    )
    contract = base_contract(
        gross_leverage=1.5,
        max_long_weight=0.8,
        max_short_weight=0.2,
        weight_tolerance=1e-6,
        annual_risk_free_rate=0.0,
        crra_gamma=2.0,
        turnover_convention="drifted_pretrade_v1",
        financing_proration="actual_calendar_days_v1",
        annualization_convention="actual_elapsed_years_v1",
        day_count_basis=365,
    )
    scored = score_strategy("future", weights, realized, ["A", "B"], contract)
    metrics = empirical_metrics(scored, contract)
    elapsed_years = 34 / 365
    factor = 2 / elapsed_years
    net_returns = scored["net_return"].to_numpy(float)
    wealth_multiple = float(np.prod(1.0 + net_returns))
    assert math.isclose(metrics["elapsed_years"], elapsed_years, rel_tol=1e-12)
    assert math.isclose(metrics["annualization_period_factor"], factor, rel_tol=1e-12)
    assert math.isclose(
        metrics["cagr"], wealth_multiple ** (1 / elapsed_years) - 1, rel_tol=1e-12
    )
    assert math.isclose(
        metrics["annual_arithmetic_return"], float(np.mean(net_returns)) * factor,
        rel_tol=1e-12,
    )


def test_legacy_annualization_remains_fixed_periods_per_year() -> None:
    weights, realized = panels(
        [[0.6, 0.4], [0.6, 0.4]],
        [[1.02, 1.02], [1.01, 1.01]],
        [31, 3],
    )
    contract = base_contract(
        gross_leverage=1.5,
        max_long_weight=0.8,
        max_short_weight=0.2,
        weight_tolerance=1e-6,
        annual_risk_free_rate=0.0,
        crra_gamma=2.0,
    )
    scored = score_strategy("legacy", weights, realized, ["A", "B"], contract)
    metrics = empirical_metrics(scored, contract)
    wealth_multiple = float(np.prod(1.0 + scored["net_return"].to_numpy(float)))
    assert math.isclose(metrics["cagr"], wealth_multiple ** 6 - 1, rel_tol=1e-12)
    assert "elapsed_years" not in metrics
