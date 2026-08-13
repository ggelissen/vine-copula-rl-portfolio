from __future__ import annotations

import ast
import json
from pathlib import Path

from publication_pipeline_draft.window_evaluation_protocol import (
    ALGORITHM_LABELS, BENCHMARK_LABELS,
)


ROOT = Path(__file__).resolve().parents[2]


def test_window_evaluator_preregisters_strong_comparators_and_robustness() -> None:
    assert len(BENCHMARK_LABELS) == 11
    assert set(ALGORITHM_LABELS) == {"td3", "ddpg", "sac", "ppo", "a2c"}
    source = (ROOT / "publication_pipeline_draft/window_evaluation_protocol.py").read_text()
    assert '"transaction_cost_sensitivity_bps": [0, 10, 25, 50]' in source
    assert '"annual_short_borrow_sensitivity_percent": [0, 3, 6, 10]' in source
    assert '"ensemble_size_sensitivity_sizes": [1, 2, 3, 5, 10]' in source
    assert '"confirmatory_claim_permitted": False' in source


def test_daily_risk_reconciles_to_monthly_common_accounting_by_construction() -> None:
    source = (ROOT / "publication_pipeline_draft/daily_mark_to_market.py").read_text()
    assert "expected_net = expected_gross * math.exp(-transaction - financing)" in source
    assert "abs(expected_net - observed_net) <= 1e-10" in source
    assert '"daily_tail_event_count"' in source


def test_extension_freeze_covers_every_new_execution_source() -> None:
    tree = ast.parse((
        ROOT / "publication_pipeline_draft/freeze_publication_extension.py"
    ).read_text())
    assignments = [node for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "SOURCES"
                           for target in node.targets)]
    assert len(assignments) == 1
    sources = set(ast.literal_eval(assignments[0].value))
    required = {
        "publication_pipeline_draft/extension_release.py",
        "publication_pipeline_draft/audit_window_rl_sweep.py",
        "publication_pipeline_draft/generate_window_policy_weights.py",
        "publication_pipeline_draft/window_evaluation_protocol.py",
        "publication_pipeline_draft/build_window_realized_panel.R",
        "publication_pipeline_draft/daily_mark_to_market.py",
        "publication_pipeline_draft/execute_window_evaluation.py",
        "publication_pipeline_draft/aggregate_walk_forward_results.py",
        "publication_pipeline_draft/config/scalability_universe_v1.json",
    }
    assert required <= sources


def test_scalability_universe_is_fixed_and_explicitly_claim_limited() -> None:
    universe = json.loads((
        ROOT / "publication_pipeline_draft/config/scalability_universe_v1.json"
    ).read_text())
    assert universe["selection_status"] == "fixed_before_return_access"
    assert len(universe["asset_order"]) == len(set(universe["asset_order"])) == 40
    assert universe["reference_asset"] == "BIL"
    assert universe["vine_truncation_level"] == 3
    assert universe["future_return_accessed_during_selection"] is False
