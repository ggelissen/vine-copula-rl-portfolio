from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from publication_pipeline_draft.synthetic_dose_protocol import (
    EXPERIMENTS, load_contract, validated_rows,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "publication_pipeline_draft/config/synthetic_dose_response_v1.json"


def make_fake_bundle(repo: Path) -> None:
    bundle = repo / "data/synthetic_dose_response_v1/vine_synthetic_100.RData"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"immutable-test-bundle")
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    indices = [int(((index - 0.5) * 1000) // 100 + 1)
               for index in range(1, 101)]
    manifest = bundle.parent / "synthetic_dose_bundle_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "protocol", "file", "sha256", "parent_file", "parent_sha256",
            "parent_pretrain_episodes", "selected_pretrain_episodes",
            "finetune_episodes", "episode_length", "selection_rule",
            "selection_indices", "evaluation_data_accessed"])
        writer.writeheader()
        writer.writerow({
            "protocol": "systematic_midpoint_100_of_1000_v1",
            "file": str(bundle), "sha256": digest, "parent_file": "parent",
            "parent_sha256": "0" * 64, "parent_pretrain_episodes": 1000,
            "selected_pretrain_episodes": 100, "finetune_episodes": 61,
            "episode_length": 24,
            "selection_rule": "floor(((i-0.5)*N)/k)+1",
            "selection_indices": ";".join(map(str, indices)),
            "evaluation_data_accessed": "FALSE"})


def test_contract_locks_explanatory_100_path_design() -> None:
    contract, _ = load_contract(CONTRACT)
    assert contract["confirmatory_claim_permitted"] is False
    assert contract["consumed_holdout_reused"] is True
    assert contract["selected_pretrain_episodes"] == 100
    assert contract["finetune_episodes"] == 61
    assert {item["experiment_id"] for item in contract["experiments"]} == EXPERIMENTS
    assert contract["design_revision_reason"]
    assert contract["selection_disclosure"]
    assert contract["benchmark_comparison_candidate"] == \
        "synthetic_100_no_policy_visible_dependence"
    assert contract["base_model"]["PRETRAIN_RANDOM_EXPLORATION_STEPS"] == "100"
    assert abs(float(contract["base_model"]["PRETRAIN_NOISE_DECAY"]) -
               0.998 ** 10) <= 1e-10


def test_job_matrix_is_two_by_ten_with_one_shared_bundle(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    path = tmp_path / "publication_pipeline_draft/config/synthetic_dose_response_v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(contract), encoding="utf-8")
    make_fake_bundle(tmp_path)
    rows, _, bundle = validated_rows(path, tmp_path, Path(
        "data/synthetic_dose_response_runs_v1"))
    assert len(rows) == 20
    assert len({(row["experiment_id"], row["seed"]) for row in rows}) == 20
    assert {row["SYNTHETIC_RETURNS_FILE"] for row in rows} == {
        "data/synthetic_dose_response_v1/vine_synthetic_100.RData"}
    assert {row["PRETRAIN_EPISODES"] for row in rows} == {"100"}
    no_visible = [
        row for row in rows
        if row["experiment_id"] ==
        "synthetic_100_no_policy_visible_dependence"
    ]
    assert len(no_visible) == 10
    assert {row["VINE_OBSERVATION_MODE"] for row in no_visible} == {"zero"}
    assert {row["VINE_FEATURE_MODE"] for row in no_visible} == {"zero"}
    assert {row["CVAR_OBSERVATION_MODE"] for row in no_visible} == {"zero"}
    assert {row["CVAR_REWARD_MODE"] for row in no_visible} == {"full"}
    assert bundle["selected_pretrain_episodes"] == "100"


def test_launcher_preserves_registered_dose_overrides() -> None:
    source = (ROOT / "run_with_config.r").read_text(encoding="utf-8")
    assert 'set_default_env("PRETRAIN_EPISODES"' in source
    assert 'set_default_env("PRETRAIN_NOISE_DECAY"' in source
    assert 'set_default_env("PRETRAIN_RANDOM_EXPLORATION_STEPS"' in source
    assert 'set_default_env("PRETRAIN_BEHAVIOR_GATE_WINDOW"' in source


def test_dose_replay_authorization_is_fail_closed() -> None:
    source = (ROOT / "evaluate_with_config.r").read_text(encoding="utf-8")
    assert '"synthetic_dose_checkpoint_audit_v1"' in source
    assert '"synthetic_dose_sweep_audit_passed"' in source
    assert 'EVAL_DOSE_CHECKPOINT_SHA256' in source


def test_dose_audit_attests_broad_and_independent_observation_modes() -> None:
    source = (ROOT /
        "publication_pipeline_draft/audit_synthetic_dose_sweep.py").read_text(
            encoding="utf-8")
    assert '"vine_observation_mode": job["VINE_OBSERVATION_MODE"]' in source
    assert '"vine_feature_mode": job["VINE_FEATURE_MODE"]' in source
    assert '"cvar_observation_mode": job["CVAR_OBSERVATION_MODE"]' in source


def test_materializer_never_reads_evaluation_data() -> None:
    source = (ROOT / "rl/materialize_synthetic_dose_bundle.r").read_text(
        encoding="utf-8")
    assert "selection_uses_returns_or_diagnostics <- FALSE" in source
    assert "evaluation_data_accessed <- FALSE" in source
    assert "parent_pretrain[selection_indices]" in source
    assert "realized_asset_gross" not in source


def test_dose_analyzer_reports_complete_and_locked_all_scopes() -> None:
    source = (ROOT /
        "publication_pipeline_draft/analyze_synthetic_dose_response.py").read_text(
            encoding="utf-8")
    assert "complete_only=True" in source
    assert "complete_only=False" in source
    assert '"locked_all_sensitivity_periods": 24' in source
    assert "synthetic_dose_locked_all_primary_contrasts.csv" in source
