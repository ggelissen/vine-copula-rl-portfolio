from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from publication_pipeline_draft.analyze_causal_results import (
    annualized_ce,
    crra_utility,
    holm_adjust,
)
from publication_pipeline_draft.causal_analysis_contract import (
    load_contract,
    materialize_plan,
)
from publication_pipeline_draft.freeze_causal_analysis_plan import SOURCES


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "publication_pipeline_draft/config/causal_analysis_contract_v1.json"


def test_causal_analysis_contract_is_complete_and_post_holdout() -> None:
    validated = load_contract(CONTRACT)
    raw = validated.raw
    assert raw["evidence_class"] == "post_holdout_explanatory"
    assert raw["confirmatory_claim_permitted"] is False
    assert len(validated.experiment_ids) == 13
    assert len(raw["expected_seeds"]) == 10
    assert len(raw["primary_component_contrasts"]) == 8
    assert len(raw["algorithm_robustness_contrasts"]) == 4
    assert raw["economics"]["ensemble_construction"] == \
        "arithmetic_mean_target_weights_then_rescore_costs"
    assert raw["economics"]["return_aggregation_forbidden"] is True
    assert raw["decision_rules"]["absence_of_significance_is_not_equivalence"] is True


def test_materialized_contrast_plan_has_all_twelve_rows(tmp_path: Path) -> None:
    output = tmp_path / "contrasts.csv"
    result = materialize_plan(CONTRACT, output)
    frame = pd.read_csv(output)
    assert result["contrast_count"] == len(frame) == 12
    assert (frame["contrast_family"].value_counts().to_dict() ==
            {"primary_component": 8, "algorithm_robustness": 4})
    assert frame["contract_sha256"].nunique() == 1


def test_crra_ce_and_holm_helpers_are_numerically_coherent() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.005])
    utility = crra_utility(returns, gamma=2.0)
    assert np.isfinite(utility).all()
    assert annualized_ce(returns, gamma=2.0, factor=12.0) < \
        annualized_ce(returns + 0.001, gamma=2.0, factor=12.0)
    adjusted = holm_adjust([0.01, 0.03, 0.20])
    assert adjusted == sorted(adjusted)
    assert all(raw <= corrected <= 1 for raw, corrected in
               zip([0.01, 0.03, 0.20], adjusted))


def test_analysis_freeze_source_closure_exists() -> None:
    assert len(SOURCES) >= 10
    missing = [relative for relative in SOURCES if not (ROOT / relative).is_file()]
    assert not missing


def test_causal_output_contract_contains_tables_and_figures() -> None:
    raw = json.loads(CONTRACT.read_text(encoding="utf-8"))
    tables = set(raw["required_outputs"]["tables"])
    figures = set(raw["required_outputs"]["figures"])
    assert "causal_primary_contrasts.csv" in tables
    assert "causal_seed_pair_effects.csv" in tables
    assert "causal_crra_effect_forest.png" in figures
    assert "causal_seed_stability.png" in figures


def test_interfaces_forbid_return_averaging_and_require_common_rescoring() -> None:
    ensemble = (ROOT / "publication_pipeline_draft/assemble_causal_policy_ensembles.py").read_text(
        encoding="utf-8")
    evaluator = (ROOT / "publication_pipeline_draft/materialize_causal_evaluation.py").read_text(
        encoding="utf-8")
    assert "stack.mean(axis=0)" in ensemble
    assert '"return_aggregation_used": False' in ensemble
    assert '"ensemble_returns_averaged": False' in evaluator
    assert '"common_realized_returns_required": True' in evaluator
