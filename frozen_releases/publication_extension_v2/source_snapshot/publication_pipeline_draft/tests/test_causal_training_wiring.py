from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_environment_masks_three_causal_channels_independently() -> None:
    source = (ROOT / "rl/rl_environment.r").read_text(encoding="utf-8")
    assert "vine_feature_mode = NULL" in source
    assert "cvar_observation_mode = NULL" in source
    assert 'cvar_reward_mode = c("full", "zero")' in source
    assert 'reward_cvar <- if (identical(private$cvar_reward_mode, "zero")) 0 else cvar' in source
    assert 'identical(private$vine_feature_mode, "zero")' in source
    assert 'identical(private$cvar_observation_mode, "zero")' in source


def test_trainer_attests_every_causal_setting() -> None:
    source = (ROOT / "rl/train_rl.r").read_text(encoding="utf-8")
    for setting in (
        "vine_feature_mode", "cvar_observation_mode", "cvar_reward_mode",
        "pretrain_data_mode", "rl_algorithm", "policy_encoder", "run_finetune",
    ):
        assert f"'{setting}':" in source
    assert "candidate.sync_targets()" in source
    assert "CHECKPOINT_PREFIX + '_full.pt'" in source
    assert "agent.record_outcome(reward, done, next_state_seq)" in source
    assert "agent.finish_episode()" in source


def test_alternative_bundles_never_read_evaluation_returns() -> None:
    source = (ROOT / "rl/generate_ablation_training_bundles.r").read_text(
        encoding="utf-8")
    assert "finetune_returns" in source
    assert "evaluation_data_accessed = FALSE" in source
    assert "load_returns" not in source
    assert "eval_period" not in source


def test_alternative_bundle_paths_preserve_frozen_mode_names() -> None:
    source = (ROOT / "rl/generate_ablation_training_bundles.r").read_text(
        encoding="utf-8")
    assert "output_files <- c(" in source
    assert "outputs <- setNames(" in source
    assert "file.path(output_root, unname(output_files)), names(output_files)" in source
    assert 'outputs[["historical_prefix_repeated"]]' in source
    assert 'outputs[["moving_block_bootstrap"]]' in source
