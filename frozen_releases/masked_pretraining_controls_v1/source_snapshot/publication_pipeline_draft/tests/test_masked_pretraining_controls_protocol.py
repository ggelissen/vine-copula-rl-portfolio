from __future__ import annotations

import json
from pathlib import Path

from publication_pipeline_draft.masked_pretraining_controls_protocol import (
    EXPERIMENTS, load_contract, validated_rows,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "publication_pipeline_draft/config/masked_pretraining_controls_v1.json"


def test_terminal_contract_is_explicitly_post_holdout_and_stops_training() -> None:
    contract, digest = load_contract(CONTRACT)
    assert len(digest) == 64
    assert contract["evidence_class"] == "post_holdout_explanatory"
    assert contract["confirmatory_claim_permitted"] is False
    assert contract["terminal_hpc_experiment"] is True
    assert "no further same-holdout neural training" in contract["stop_rule"]


def test_exact_masked_architecture_and_budget_are_matched() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    base = contract["base_model"]
    assert base["RL_ALGORITHM"] == "td3"
    assert base["POLICY_ENCODER"] == "lstm"
    assert base["VINE_OBSERVATION_MODE"] == "zero"
    assert base["VINE_FEATURE_MODE"] == "zero"
    assert base["CVAR_OBSERVATION_MODE"] == "zero"
    assert base["CVAR_REWARD_MODE"] == "full"
    assert base["PRETRAIN_EPISODES"] == "1000"
    assert base["PRETRAIN_RANDOM_EXPLORATION_STEPS"] == "1000"
    assert base["PRETRAIN_NOISE_DECAY"] == "0.998"


def test_only_historical_prefix_and_moving_block_controls_are_registered() -> None:
    contract, _ = load_contract(CONTRACT)
    assert {item["experiment_id"] for item in contract["experiments"]} == EXPERIMENTS
    modes = {item.get("overrides", {}).get(
        "PRETRAIN_DATA_MODE", contract["base_model"]["PRETRAIN_DATA_MODE"])
        for item in contract["experiments"]}
    assert modes == {"historical_prefix_repeated", "moving_block_bootstrap"}


def test_job_matrix_has_twenty_matched_seed_policies_without_data_access() -> None:
    rows, _, bundles = validated_rows(
        CONTRACT, ROOT, Path("data/masked_pretraining_control_runs_v1"),
        validate_inputs=False)
    assert bundles == []
    assert len(rows) == 20
    assert len({(row["experiment_id"], row["seed"]) for row in rows}) == 20
    assert {row["seed"] for row in rows} == set(range(20261001, 20261011))
    assert all(row["pretrain_episode_presentations"] == 1000 for row in rows)
    assert all(row["VINE_FEATURE_MODE"] == "zero" for row in rows)
    assert all(row["CVAR_OBSERVATION_MODE"] == "zero" for row in rows)


def test_primary_family_closes_the_generator_value_confound() -> None:
    contract, _ = load_contract(CONTRACT)
    candidate = contract["candidate_experiment_id"]
    assert {(item["candidate"], item["comparator"])
            for item in contract["primary_contrasts"]} == {
        (candidate, "masked_historical_prefix_1000_presentations"),
        (candidate, "masked_moving_block_bootstrap_1000_presentations"),
    }
    assert contract["inference"]["multiplicity"].startswith("Holm_across_the_two")


def test_audit_enforces_identical_actions_updates_and_checkpoint_metadata() -> None:
    source = (ROOT / "publication_pipeline_draft/"
              "audit_masked_pretraining_controls.py").read_text(encoding="utf-8")
    assert "expected_pretrain_actions = 1000" in source
    assert "expected_full_actions" in source
    assert "len(update_counts) == 1" in source
    assert '"vine_feature_mode": "zero"' in source
    assert '"cvar_observation_mode": "zero"' in source
    assert '"experiment_protocol": "terminal_masked_pretraining_controls_v1"' in source


def test_replay_is_audited_weights_only_and_analysis_reuses_candidate_hash() -> None:
    runner = (ROOT / "publication_pipeline_draft/"
              "run_masked_pretraining_controls.py").read_text(encoding="utf-8")
    replay = (ROOT / "publication_pipeline_draft/"
              "generate_masked_pretraining_control_weights.py").read_text(
                  encoding="utf-8")
    analysis = (ROOT / "publication_pipeline_draft/"
                "analyze_masked_pretraining_controls.py").read_text(encoding="utf-8")
    assert '"EVAL_WEIGHTS_ONLY": "true"' in replay
    assert '"synthetic_dose_checkpoint_audit_v1"' in replay
    assert "sha256(bundle) == row[\"bundle_sha256\"].lower()" in runner
    assert "candidate_weight_manifest_sha256" in analysis
    assert "two_primary_generator_value_contrasts" in analysis
    assert "same_holdout_neural_training_complete" in analysis
