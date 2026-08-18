from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from publication_pipeline_draft.analyze_focused_walk_forward import analyze
from publication_pipeline_draft.focused_window_training_protocol import (
    materialize, validate_protocol,
)
from publication_pipeline_draft.focused_seven_asset_panel import ASSETS
from publication_pipeline_draft.focused_seven_asset_panel import (
    materialize as materialize_panel,
)
from publication_pipeline_draft.focused_walk_forward_windows import (
    materialize as materialize_windows,
)
from publication_pipeline_draft.run_focused_window_sweep import (
    FocusedSweepError, attested_episode_counts,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "publication_pipeline_draft/config/focused_walk_forward_mechanisms_v1.json")
PROGRAM = ROOT / (
    "publication_pipeline_draft/config/publication_research_program_v2.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_window(tmp_path: Path) -> Path:
    root = tmp_path / "window_input"
    root.mkdir()
    returns = root / "window_daily_log_returns.csv"
    returns.write_text("date," + ",".join(ASSETS) + "\n",
                       encoding="utf-8")
    manifest = {
        "release_status": "frozen_window_return_input_no_confirmation",
        "confirmatory_claim_permitted": False,
        "panel_id": "original_seven_asset_panel",
        "window_id": "retrospective_original_7asset_expanding_24m_v1_w01",
        "evidence_class": "retrospective_walk_forward",
        "asset_count": 7,
        "reference_asset": "GOLD",
        "reference_asset_index_1based": 7,
        "vine_truncation_level": 6,
        "expected_evaluation_periods": 24,
        "return_file_sha256": digest(returns),
    }
    manifest_path = root / "return_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    note = root / "READ_ONLY_WINDOW_INPUT.txt"
    note.write_text("development\n", encoding="utf-8")
    (root / "CONTENTS.sha256").write_text(
        "\n".join(f"{digest(path)}  {path.name}"
                  for path in (note, manifest_path, returns)) + "\n",
        encoding="ascii")
    return root


def test_focused_protocol_is_three_by_five_and_nonconfirmatory() -> None:
    protocol, digest_value = validate_protocol(PROTOCOL)
    assert len(digest_value) == 64
    assert protocol["confirmatory_claim_permitted"] is False
    assert len(protocol["experiments"]) == 3
    assert len(protocol["seeds"]) == 5
    assert len(protocol["contrasts"]) == 2


def test_original_panel_freezes_exactly_two_registered_windows(
        tmp_path: Path) -> None:
    source = ROOT / "data/portfolio_B_7assets_2013.csv"
    panel_root = tmp_path / "panel"
    panel = materialize_panel(source, panel_root)
    assert panel["panel_id"] == "original_seven_asset_panel"
    assert panel["evidence_class"] == "retrospective_walk_forward"
    assert panel["contains_previously_consumed_holdout"] is True
    assert panel["fresh_confirmatory_data_accessed"] is False
    window_root = tmp_path / "windows"
    window = materialize_windows(
        PROTOCOL,
        panel_root / "development_monthly_asset_gross.csv",
        panel_root / "development_panel_manifest.json",
        window_root,
    )
    assert window["window_count"] == 2
    assert window["windows_nonoverlapping"] is True
    assert window["confirmatory_claim_permitted"] is False
    with (window_root / "window_schedule.csv").open(
            newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["test_end"] <= rows[1]["test_start"]
    assert all(row["confirmatory_claim_permitted"] == "False" for row in rows)


def test_focused_window_contract_materializes_15_jobs(tmp_path: Path) -> None:
    window = make_window(tmp_path)
    output = tmp_path / "contract"
    result = materialize(ROOT, PROGRAM, PROTOCOL, window,
                         Path("data/focused"), output)
    with (output / "focused_window_jobs.csv").open(
            newline="", encoding="utf-8") as stream:
        jobs = list(csv.DictReader(stream))
    assert result["job_count"] == len(jobs) == 15
    assert len({row["experiment_id"] for row in jobs}) == 3
    assert all(row["RL_ALGORITHM"] == "td3" for row in jobs)
    assert all(row["PRETRAIN_BEHAVIOR_GATE_MODE"] == "report_only"
               for row in jobs)
    assert result["asset_count"] == 7
    assert result["reference_asset"] == "GOLD"
    assert result["vine_truncation_level"] == 6


def test_focused_protocol_keeps_six_financial_benchmarks() -> None:
    protocol, _ = validate_protocol(PROTOCOL)
    assert protocol["financial_benchmarks"] == [
        "equal_weight", "shrinkage_mean_variance", "dcc_garch",
        "static_vine", "rolling_vine", "dynamic_nn_vine",
    ]
    assert protocol["benchmark_candidate_experiment_id"] == \
        "zero_vine_features_keep_cvar_observation"
    assert protocol["benchmark_multiplicity_control"] == \
        "holm_across_six_financial_benchmarks"


def test_focused_analysis_uses_windows_not_seeds_as_market_sample(
        tmp_path: Path) -> None:
    protocol, _ = validate_protocol(PROTOCOL)
    rows = []
    start = date(2018, 1, 31)
    for window_number in range(2):
        window_id = f"w{window_number + 1}"
        for experiment_number, experiment in enumerate(protocol["experiments"]):
            experiment_id = experiment["experiment_id"]
            for level, seeds in (("ensemble", [None]),
                                 ("seed", protocol["seeds"])):
                for seed in seeds:
                    strategy_id = (f"{experiment_id}_ensemble" if seed is None
                                   else f"{experiment_id}_seed_{seed}")
                    for period in range(20):
                        decision = start + timedelta(
                            days=(window_number * 24 + period) * 31)
                        base = 0.01 + 0.0001 * period
                        # Make the reference mildly better than both controls.
                        value = base + (0.001 if experiment_number == 0 else 0.0)
                        if seed is not None:
                            value += (int(seed) % 5) * 1e-5
                        rows.append({
                            "window_id": window_id,
                            "experiment_id": experiment_id,
                            "strategy_level": level,
                            "strategy_id": strategy_id,
                            "seed": "" if seed is None else seed,
                            "decision_date": decision.isoformat(),
                            "holding_end_date": (decision + timedelta(days=30)).isoformat(),
                            "net_return": value,
                            "gross_return": value + 0.0001,
                            "turnover": 0.20,
                            "transaction_cost": 0.0001,
                            "financing_cost": 0.00002,
                            "gross_exposure": 1.10,
                            "short_notional": 0.05,
                            "is_complete_period": True,
                        })
        for benchmark_number, benchmark_id in enumerate(
                protocol["financial_benchmarks"]):
            for period in range(20):
                decision = start + timedelta(
                    days=(window_number * 24 + period) * 31)
                rows.append({
                    "window_id": window_id,
                    "experiment_id": benchmark_id,
                    "strategy_level": "benchmark",
                    "strategy_id": benchmark_id,
                    "seed": "",
                    "decision_date": decision.isoformat(),
                    "holding_end_date": (decision + timedelta(days=30)).isoformat(),
                    "net_return": 0.009 - benchmark_number * 1e-5,
                    "gross_return": 0.0091 - benchmark_number * 1e-5,
                    "turnover": 0.10,
                    "transaction_cost": 0.0001,
                    "financing_cost": 0.0,
                    "gross_exposure": 1.0,
                    "short_notional": 0.0,
                    "is_complete_period": True,
                })
    panel = tmp_path / "periods.csv"
    pd.DataFrame(rows).to_csv(panel, index=False)
    output = tmp_path / "analysis"
    result = analyze(PROTOCOL, panel, output)
    assert result["window_count"] == 2
    assert result["seed_inference_scope"] == "optimization_variability_only"
    contrasts = pd.read_csv(output / "focused_walk_forward_contrasts.csv")
    assert len(contrasts) == 2
    assert np.isfinite(contrasts["annualized_ce_difference"]).all()
    assert (contrasts["annualized_ce_difference"] > 0).all()
    derived = pd.read_csv(
        output / "focused_walk_forward_derived_contrasts.csv")
    assert len(derived) == 1
    assert derived["analysis_status"].iloc[0] == \
        "post_hoc_derived_not_in_preregistered_multiplicity_family"
    benchmarks = pd.read_csv(
        output / "focused_walk_forward_benchmark_comparisons.csv")
    assert len(benchmarks) == 6
    assert np.isfinite(benchmarks["annualized_ce_difference"]).all()
    assert (benchmarks["annualized_ce_difference"] > 0).all()
    window_metrics = pd.read_csv(
        output / "focused_walk_forward_window_metrics.csv")
    assert len(window_metrics) == 48
    assert set(window_metrics["strategy_level"]) == {
        "seed", "ensemble", "benchmark"}
    pooled_metrics = pd.read_csv(
        output / "focused_walk_forward_pooled_metrics.csv")
    assert len(pooled_metrics) == 24
    assert set(pooled_metrics["window_count"]) == {2}
    window_effects = pd.read_csv(
        output / "focused_walk_forward_window_effects.csv")
    assert len(window_effects) == 18
    assert set(window_effects["comparison_family"]) == {
        "mechanism", "derived_mechanism", "financial_benchmark"}


def test_focused_replay_authorization_is_fail_closed() -> None:
    source = (ROOT / "evaluate_with_config.r").read_text(encoding="utf-8")
    assert '"focused_checkpoint_audit_v1"' in source
    assert 'audit_manifest$job_count), 15L' in source
    assert 'audit_manifest$experiment_count), 3L' in source
    assert 'audit_manifest$seeds_per_experiment), 5L' in source
    assert "Audited focused replay is restricted to weights-only" in source


def test_focused_common_accounting_is_drifted_and_keeps_seeds() -> None:
    source = (ROOT / "publication_pipeline_draft/score_focused_window.py").read_text(
        encoding="utf-8")
    assert '"turnover_convention": "drifted_pretrade_v1"' in source
    assert '"seed_strategy_count": 15' in source
    assert '"ensemble_strategy_count": 3' in source
    assert '"benchmark_strategy_count": 6' in source
    assert "score_strategy(strategy_id, weight, realized, assets, contract)" in source


def test_focused_result_freezer_requires_all_thirty_checkpoints() -> None:
    source = (ROOT /
        "publication_pipeline_draft/freeze_focused_walk_forward_results.py").read_text(
            encoding="utf-8")
    assert '"checkpoint_count": 30' in source
    assert '"financial_benchmark_count_per_window": 6' in source
    assert '"contains_previously_consumed_holdout": True' in source
    assert '"confirmatory_claim_permitted": False' in source
    assert "Checkpoint changed after audit" in source


def test_focused_sweep_uses_attested_window_episode_counts() -> None:
    source = (ROOT /
        "publication_pipeline_draft/run_focused_window_sweep.py").read_text(
            encoding="utf-8")
    assert "environment.update(episode_counts)" in source
    assert '"--preflight-only"' in source
    assert '"focused_window_sweep_preflight_passed"' in source
    assert attested_episode_counts({
        "pretrain_episodes": 1000,
        "finetune_episodes": 18,
    }) == {
        "PRETRAIN_EPISODES": "1000",
        "FINETUNE_EPISODES": "18",
    }
    for invalid in (
            {},
            {"pretrain_episodes": 1000, "finetune_episodes": 0},
            {"pretrain_episodes": "bad", "finetune_episodes": 18}):
        try:
            attested_episode_counts(invalid)
        except FocusedSweepError:
            pass
        else:
            raise AssertionError("Invalid episode counts did not fail closed.")


def test_short_window_uses_fixed_one_pass_all_history_finetuning() -> None:
    trainer = (ROOT / "rl/train_rl.r").read_text(encoding="utf-8")
    assert "has_purged_finetune_validation <- selection_fit_count >= 1L" in trainer
    assert "finetune_max_selection_passes != 1L" in trainer
    assert "fixed_one_pass_all_history_no_validation_short_window" in trainer
    assert "Purged validation diagnostic skipped for short history" in trainer
    assert "episodes = all_count * best_pass" in trainer
    assert "Too few historical episodes for purged fine-tuning validation." not in trainer


def test_focused_benchmark_contract_has_fail_closed_objective_tolerance() -> None:
    contract = json.loads((ROOT /
        "publication_pipeline_draft/config/benchmark_contract_v4.json").read_text(
            encoding="utf-8"))
    solver = (ROOT / "publication_pipeline_draft/benchmark_weights.R").read_text(
        encoding="utf-8")
    assert contract["optimizer_ftol_rel"] == 1e-8
    assert contract["optimizer_ftol_abs"] == 1e-7
    assert contract["optimizer_xtol_rel"] == 1e-9
    assert contract["optimizer_maxeval"] == 4000
    assert contract["optimizer_allowed_convergence_codes"] == [1, 2, 3, 4]
    assert "ftol_rel = as.numeric(contract$optimizer_ftol_rel %||% 0)" in solver
    assert "ftol_abs = as.numeric(contract$optimizer_ftol_abs %||% 0)" in solver
    assert "Code 5 is not accepted" in solver
    assert "kinds <- kinds[methods]" in solver


def test_sampling_aware_synthetic_gate_is_guardrailed_and_auditable() -> None:
    helper = (ROOT / "helper/synthetic_fidelity.r").read_text(encoding="utf-8")
    generator = (ROOT / "rl/synthetic_returns.r").read_text(encoding="utf-8")
    revalidator = (ROOT / "rl/revalidate_synthetic_bundle.r").read_text(
        encoding="utf-8")
    preparer = (ROOT /
        "publication_pipeline_draft/prepare_window_training_data.py").read_text(
            encoding="utf-8")
    assert "marginal_guardrail_pass" in helper
    assert "correlation_guardrail_pass" in helper
    assert "episode_cluster_correlation_intervals" in helper
    assert 'diagnostic_gate_protocol = "sampling_aware_guardrailed_v2"' in generator
    assert "post_generation_statistical_revision_without_resimulation" in revalidator
    assert '"synthetic_returns_regenerated": False' in preparer
    assert "--adopt-existing-revalidated" in preparer
