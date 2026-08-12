from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_return_loader_supports_only_explicit_frozen_external_inputs() -> None:
    source = (ROOT / "helper/load_data.r").read_text(encoding="utf-8")
    assert 'Sys.getenv("RETURNS_DATA_FILE"' in source
    assert 'Sys.getenv("RETURNS_DATA_KIND", "adjusted_levels")' in source
    assert '"frozen_window_return_input_no_confirmation"' in source
    assert "Daily return file hash does not match" in source
    assert "Daily return columns/order do not match" in source
    assert "validate_return_evaluation_contract" in source


def test_training_never_hard_codes_original_panel_in_provenance() -> None:
    source = (ROOT / "rl/train_rl.r").read_text(encoding="utf-8")
    manifest_call = source[source.index("write_run_manifest"):source.index(
        "cat(sprintf(\"Run mode", source.index("write_run_manifest"))]
    assert 'Sys.getenv("RETURNS_DATA_FILE"' in manifest_call
    assert 'Sys.getenv("RETURNS_DATA_MANIFEST"' in manifest_call
    assert "source-data hash does not match the active panel" in source
    assert "asset names/order do not match the active panel" in source


def test_launchers_preserve_scheduler_window_overrides() -> None:
    trainer = (ROOT / "run_with_config.r").read_text(encoding="utf-8")
    evaluator = (ROOT / "evaluate_with_config.r").read_text(encoding="utf-8")
    for source in (trainer, evaluator):
        assert 'set_default_env("RETURNS_DATA_FILE"' in source
        assert 'set_default_env("RETURNS_DATA_KIND"' in source
        assert 'set_default_env("RETURNS_DATA_MANIFEST"' in source
        assert 'set_default_env("REF_COL"' in source
        assert 'set_default_env("NN_VINE_MODEL_DIR"' in source


def test_evaluator_replays_independent_causal_modes() -> None:
    source = (ROOT / "rl/evaluate_rl.r").read_text(encoding="utf-8")
    assert 'vine_feature_mode <- Sys.getenv("VINE_FEATURE_MODE"' in source
    assert 'cvar_observation_mode <- Sys.getenv("CVAR_OBSERVATION_MODE"' in source
    assert 'cvar_reward_mode <- Sys.getenv("CVAR_REWARD_MODE"' in source
    assert "environment_arguments$vine_feature_mode" in source
    assert '"rl/policy_inference_server_v2.py"' in source
