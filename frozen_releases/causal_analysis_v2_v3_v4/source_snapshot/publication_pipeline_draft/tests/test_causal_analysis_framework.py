from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from publication_pipeline_draft.analyze_causal_results import (
    annualized_ce,
    crra_utility,
    holm_adjust,
    training_gate_diagnostics,
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
    assert "hpc/finalize_causal_evaluation_v4.sh" in SOURCES


def test_analysis_freezer_treats_training_releases_as_immutable_history() -> None:
    source = (ROOT /
        "publication_pipeline_draft/freeze_causal_analysis_plan.py").read_text(
            encoding="utf-8")
    assert "extension = verify_frozen_extension_integrity" in source
    assert "extension = verify_extension_release" not in source
    assert "snapshot and hash the current analysis SOURCES independently" in source


def test_causal_output_contract_contains_tables_and_figures() -> None:
    raw = json.loads(CONTRACT.read_text(encoding="utf-8"))
    tables = set(raw["required_outputs"]["tables"])
    figures = set(raw["required_outputs"]["figures"])
    assert "causal_primary_contrasts.csv" in tables
    assert "causal_seed_pair_effects.csv" in tables
    assert "causal_training_gate_diagnostics.csv" in tables
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


def test_causal_interface_preserves_report_only_gate_evidence() -> None:
    materializer = (ROOT /
        "publication_pipeline_draft/materialize_causal_evaluation.py").read_text(
            encoding="utf-8")
    exporter = (ROOT /
        "publication_pipeline_draft/export_causal_period_panel.py").read_text(
            encoding="utf-8")
    analyzer = (ROOT /
        "publication_pipeline_draft/analyze_causal_results.py").read_text(
            encoding="utf-8")
    for field in ("protocol_eligibility_pass", "behavior_gate_pass",
                  "behavior_gate_mode", "behavior_gate_failed_metrics",
                  "operational_source"):
        assert field in materializer
        assert field in exporter
    assert "intent_to_train_no_economic_gate_selection" in analyzer
    assert "causal_policy_weight_replay_complete" in materializer
    assert "causal_weight_ensembles_complete" in materializer
    assert "all_behavior_gate_enforcement_valid" in materializer


def test_three_revision_runbook_uses_only_current_finalization_paths() -> None:
    runbook = (ROOT / "publication_pipeline_draft/CAUSAL_ANALYSIS_RUNBOOK.md").read_text(
        encoding="utf-8")
    assert "causal_jobs_v2_v3_v4_merged.csv" in runbook
    assert "causal_sweep_audit_v2_v3_v4" in runbook
    assert "causal_results_v2_v3_v4.tar.gz" in runbook
    assert "hpc/finalize_causal_evaluation_v4.sh" in runbook


def test_training_gate_table_separates_eligibility_from_economic_warnings() -> None:
    rows = []
    for experiment_index in range(13):
        for seed_index in range(10):
            report_only = experiment_index == 0 and seed_index < 3
            rows.append({
                "experiment_id": f"experiment_{experiment_index}",
                "strategy_id": f"strategy_{experiment_index}_{seed_index}",
                "strategy_level": "seed", "seed": 20261001 + seed_index,
                "protocol_eligibility_pass": True,
                "behavior_gate_pass": not report_only,
                "behavior_gate_mode": "report_only" if report_only else "strict",
                "behavior_gate_failed_metrics": "mean_turnover" if report_only else "",
                "operational_source": "v4_report_only" if report_only else "v2_strict",
            })
    table = training_gate_diagnostics(pd.DataFrame(rows))
    assert len(table) == 13
    assert table["protocol_eligible_count"].sum() == 130
    assert table["economic_warning_count"].sum() == 3
    assert table["report_only_included_count"].sum() == 3


def test_final_causal_release_is_self_contained_for_rescoring() -> None:
    freezer = (ROOT / "publication_pipeline_draft/freeze_causal_results.py").read_text(
        encoding="utf-8")
    for artifact in ("analysis_plan_release", "checkpoint_audit",
                     "policy_weights", "policy_ensembles",
                     "common_accounting", "analysis_results"):
        assert artifact in freezer
    assert '"self_contained_result_evidence": True' in freezer
    assert '"realized_asset_return_panel_included": True' in freezer
    assert '"neural_checkpoints_included": False' in freezer
