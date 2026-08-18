from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from publication_pipeline_draft.run_terminal_robustness import (
    daily_replay,
    friction_surface,
    moving_block_indices,
    stationary_indices,
)
from publication_pipeline_draft.terminal_robustness_protocol import load_contract


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "publication_pipeline_draft/config/terminal_robustness_v1.json"


def contract_value() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_terminal_contract_prohibits_training_selection_and_new_confirmation() -> None:
    value = load_contract(CONTRACT)
    assert value["policy_retraining_permitted"] is False
    assert value["model_selection_permitted"] is False
    assert value["confirmatory_claim_created"] is False
    assert "No additional same-holdout policy training" in value["stop_rule"]
    assert {item["evidence_class"] for item in value["sources"]} == {
        "frozen_primary_evaluation",
        "post_holdout_explanatory",
        "retrospective_walk_forward",
    }


def test_terminal_contract_registers_daily_friction_and_resampling_checks() -> None:
    value = load_contract(CONTRACT)
    assert value["daily_risk"]["tail_probabilities"] == [0.95, 0.99]
    assert value["economics"]["transaction_cost_bps_grid"] == [0, 10, 25, 50]
    assert value["economics"]["annual_short_borrow_percent_grid"] == [0, 3, 6, 10]
    assert value["inference"]["moving_block_lengths"] == [1, 2, 3, 4, 6]
    assert value["inference"]["stationary_expected_block_lengths"] == [2, 3, 6, 12]
    assert value["inference"]["bootstrap_replications"] == 50000


def test_resampling_indices_are_deterministic_and_circular() -> None:
    first = moving_block_indices(np.random.default_rng(17), 100, 7, 3)
    second = moving_block_indices(np.random.default_rng(17), 100, 7, 3)
    assert np.array_equal(first, second)
    assert first.shape == (100, 7)
    assert first.min() >= 0 and first.max() < 7
    stationary = stationary_indices(np.random.default_rng(19), 100, 7, 3)
    assert stationary.shape == (100, 7)
    assert stationary.min() >= 0 and stationary.max() < 7


def synthetic_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    assets = contract_value()["asset_order"]
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    gross_values = np.array([
        [1.01, 1.00, 0.99, 1.02, 1.00, 1.01, 0.995],
        [1.00, 1.02, 1.01, 0.99, 1.005, 1.00, 1.01],
        [0.995, 1.00, 1.01, 1.00, 1.01, 0.99, 1.00],
        [1.01, 0.99, 1.00, 1.01, 1.00, 1.02, 1.005],
    ])
    daily = pd.DataFrame(gross_values, columns=assets)
    daily.insert(0, "date", dates)
    weights = np.repeat(1 / 7, 7)
    asset_month = gross_values.prod(axis=0)
    gross = 1 + float(np.dot(weights, asset_month - 1))
    transaction = 0.001
    financing = 0.0002
    row = {
        "source_id": "test_source", "evidence_class": "post_holdout_explanatory",
        "claim_scope": "unit_test",
        "canonical_strategy_id": "test_source::test_strategy",
        "strategy_id": "test_strategy", "window_id": "test_window",
        "decision_date": pd.Timestamp("2025-01-01"),
        "holding_end_date": pd.Timestamp("2025-01-07"),
        "trading_days": 4, "is_complete_period": True,
        "gross_return": gross - 1, "net_return": gross * np.exp(
            -transaction - financing) - 1,
        "turnover": 1.0, "transaction_cost": transaction,
        "financing_cost": financing, "short_notional": 0.0,
        "cash_borrow_notional": 0.0, "gross_exposure": 1.0,
        "net_exposure": 1.0,
    }
    row.update({f"w_{asset}": weights[index] for index, asset in enumerate(assets)})
    contract = contract_value()
    contract["economics"]["transaction_cost_bps_grid"] = [0, 10]
    contract["economics"]["annual_short_borrow_percent_grid"] = [0, 3]
    contract["economics"]["annual_cash_borrow_percent_grid"] = [0, 2]
    return pd.DataFrame([row]), daily, contract


def test_daily_replay_exactly_reconciles_frozen_monthly_accounting() -> None:
    panel, daily, contract = synthetic_panel()
    replay, reconciliation = daily_replay(panel, daily, contract)
    assert len(replay) == 4
    assert reconciliation["reconciliation_pass"].all()
    assert reconciliation[["gross_error", "net_error"]].abs().to_numpy().max() < 1e-12
    assert np.isclose(replay["transaction_log_cost"].sum(), 0.001)
    assert np.isclose(replay["financing_log_cost"].sum(), 0.0002)


def test_friction_surface_is_fixed_weight_rescoring() -> None:
    panel, _, contract = synthetic_panel()
    result = friction_surface(panel, contract)
    # Two scopes and 2 x 2 x 2 economic grid points.
    assert len(result) == 16
    assert set(result["transaction_cost_bps_one_way"]) == {0, 10}
    zero = result[(result["scope"] == "complete_periods") &
                  (result["transaction_cost_bps_one_way"] == 0) &
                  (result["annual_short_borrow_percent"] == 0) &
                  (result["annual_cash_borrow_percent"] == 0)].iloc[0]
    ten = result[(result["scope"] == "complete_periods") &
                 (result["transaction_cost_bps_one_way"] == 10) &
                 (result["annual_short_borrow_percent"] == 0) &
                 (result["annual_cash_borrow_percent"] == 0)].iloc[0]
    assert ten["total_return"] < zero["total_return"]


def test_hpc_launcher_exposes_fail_closed_stages() -> None:
    source = (ROOT / "hpc/run_terminal_robustness_v1.sh").read_text(encoding="utf-8")
    for stage in ("validate", "inputs", "freeze", "run", "status", "verify",
                  "cleanroom", "finalize"):
        assert f"{stage})" in source
    assert "OMP_NUM_THREADS=1" in source
    assert "require_absent \"$RESULTS\"" in source
