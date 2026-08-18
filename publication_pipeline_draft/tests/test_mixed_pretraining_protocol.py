from __future__ import annotations

import json
from pathlib import Path

from publication_pipeline_draft.mixed_pretraining_protocol import load_contract

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "publication_pipeline_draft/config/mixed_pretraining_response_v1.json"


def test_contract_is_terminal_post_holdout_and_outcome_independent() -> None:
    contract, digest = load_contract(CONTRACT)
    assert len(digest) == 64
    assert contract["evidence_class"] == "post_holdout_explanatory"
    assert contract["confirmatory_claim_permitted"] is False
    assert contract["terminal_same_holdout_training"] is True
    assert contract["protocol_deviation_from_prior_stop_rule"] is True
    assert "explicit protocol deviation" in contract["prior_stop_rule_disclosure"]
    assert "reported regardless" in contract["outcome_independent_rule"]
    assert "final same-holdout" in contract["stop_rule"]


def test_mixed_geometry_and_four_arm_estimand_are_exact() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert (contract["synthetic_unique_episode_count"],
            contract["historical_unique_episode_count"],
            contract["mixed_unique_episode_count"],
            contract["pretrain_episode_presentations"],
            contract["finetune_episodes"]) == (100, 61, 161, 1000, 61)
    assert (contract["synthetic_episode_presentations"],
            contract["historical_episode_presentations"]) == (621, 379)
    assert len(contract["experiments"]) == 1
    assert len(contract["seeds"]) == 10
    assert len(contract["comparison_arms"]) == 4
    assert len(contract["primary_contrasts"]) == 3
    assert contract["economic_guardrails"] == {
        "maximum_mean_monthly_turnover_increase": 0.10,
        "maximum_mean_gross_exposure_increase": 0.10,
    }
    assert all(item["candidate"] ==
               "mixed_pretraining_plus_historical_finetuning"
               for item in contract["primary_contrasts"])


def test_materializer_is_deterministic_and_evaluation_blind() -> None:
    source = (ROOT / "rl/materialize_mixed_pretraining_bundle.r").read_text(
        encoding="utf-8")
    assert "synthetic_position <- (seq_len(100L) - 0.5) / 100L" in source
    assert "historical_position <- (seq_len(61L) - 0.5) / 61L" in source
    assert "presentation_unique_index <- rep(seq_len(161L), length.out = 1000L)" in source
    assert "metadata$evaluation_data_accessed <- FALSE" in source
    assert "metadata$selection_uses_returns_or_diagnostics <- FALSE" in source


def test_trainer_accepts_and_validates_mixed_mode() -> None:
    source = (ROOT / "rl/train_rl.r").read_text(encoding="utf-8")
    assert '"mixed_historical_synthetic"' in source
    assert 'mixed_historical_synthetic = "historical_synthetic_mixture"' in source
    assert "metadata$mixed_pretraining_bundle" in source
    assert "metadata$mixed_episode_presentations" in source
    assert "registered pretraining-presentation count" in source


def test_mixed_replay_authorization_is_fail_closed() -> None:
    source = (ROOT / "evaluate_with_config.r").read_text(encoding="utf-8")
    assert '"mixed_pretraining_checkpoint_audit_v1"' in source
    assert 'checkpoint_model %in% c("full", "pretrained")' in source
    assert '"mixed_pretraining_comparison_audit_passed"' in source
    assert 'audit_table$checkpoint_model == checkpoint_model' in source
    assert "Checkpoint is not uniquely authorized by the mixed-pretraining audit" in source


def test_analysis_writes_table_figure_and_three_contrasts() -> None:
    source = (ROOT /
        "publication_pipeline_draft/analyze_mixed_pretraining_response.py").read_text(
            encoding="utf-8")
    assert "mixed_pretraining_four_arm_metrics.csv" in source
    assert "table_mixed_pretraining_four_arm.tex" in source
    assert "figure_mixed_pretraining_four_arm.tex" in source
    assert "mixed_pretraining_primary_contrasts.csv" in source
    assert "same_holdout_further_tuning_authorized" in source
    assert "Holm" not in source or "build_contrasts" in source


def test_hpc_script_only_trains_ten_new_policies() -> None:
    source = (ROOT / "hpc/run_mixed_pretraining_response_v1.sh").read_text(
        encoding="utf-8")
    assert "run_mixed_pretraining_sweep" in source
    assert "synthetic_presentation_policy_weight_manifest.csv" in source
    assert "masked_pretraining_control_weight_manifest.csv" in source
    assert "checkpoint-archive" in source
    assert "finalize" in source
