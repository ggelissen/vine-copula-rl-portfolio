from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from publication_pipeline_draft.synthetic_presentation_protocol import (
    EXPERIMENTS, load_contract, validated_rows,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / (
    "publication_pipeline_draft/config/synthetic_presentation_response_v2.json")


def make_fake_bundles(repo: Path) -> None:
    source = repo / "data/synthetic_dose_response_v1/vine_synthetic_100.RData"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"immutable-100-path-source")
    output = repo / ("data/synthetic_presentation_response_v2/"
                     "vine_synthetic_100_unique_1000_presentations.RData")
    output.parent.mkdir(parents=True)
    output.write_bytes(b"immutable-100-by-10-presentation-bundle")
    manifest = output.parent / "synthetic_presentation_bundle_manifest.csv"
    fields = [
        "protocol", "file", "sha256", "source_100_path_file",
        "source_100_path_sha256", "parent_pretrain_episodes",
        "synthetic_unique_episode_count", "synthetic_episode_presentations",
        "repetition_count", "finetune_episodes", "episode_length",
        "presentation_rule", "source_indices_per_pass",
        "selection_uses_returns_or_diagnostics", "evaluation_data_accessed",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerow({
            "protocol": "ordered_10_passes_of_systematic_midpoint_100_v2",
            "file": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "source_100_path_file": str(source),
            "source_100_path_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "parent_pretrain_episodes": 1000,
            "synthetic_unique_episode_count": 100,
            "synthetic_episode_presentations": 1000, "repetition_count": 10,
            "finetune_episodes": 61, "episode_length": 24,
            "presentation_rule": "ten_ordered_complete_passes",
            "source_indices_per_pass": ";".join(map(str, range(1, 101))),
            "selection_uses_returns_or_diagnostics": "FALSE",
            "evaluation_data_accessed": "FALSE",
        })


def test_contract_locks_identification_geometry_and_claim_class() -> None:
    contract, _ = load_contract(CONTRACT)
    assert contract["evidence_class"] == "post_holdout_explanatory"
    assert contract["confirmatory_claim_permitted"] is False
    assert contract["synthetic_unique_episode_count"] == 100
    assert contract["synthetic_episode_presentations"] == 1000
    assert contract["repetition_count"] == 10
    assert len(contract["primary_contrasts"]) == 4
    assert {item["experiment_id"] for item in contract["experiments"]} == EXPERIMENTS
    assert contract["base_model"]["PRETRAIN_EPISODES"] == "1000"
    assert contract["base_model"]["PRETRAIN_RANDOM_EXPLORATION_STEPS"] == "1000"
    assert contract["base_model"]["PRETRAIN_NOISE_DECAY"] == "0.998"


def test_jobs_are_two_by_ten_and_preserve_100_unique_1000_presentations(
        tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    path = tmp_path / (
        "publication_pipeline_draft/config/synthetic_presentation_response_v2.json")
    path.parent.mkdir(parents=True); path.write_text(json.dumps(contract), encoding="utf-8")
    make_fake_bundles(tmp_path)
    source = tmp_path / "data/synthetic_dose_response_v1/vine_synthetic_100.RData"
    contract["source_100_path_bundle_sha256"] = hashlib.sha256(
        source.read_bytes()).hexdigest()
    path.write_text(json.dumps(contract), encoding="utf-8")
    rows, _, bundle = validated_rows(
        path, tmp_path, Path("data/synthetic_presentation_response_runs_v2"))
    assert len(rows) == 20
    assert len({(row["experiment_id"], row["seed"]) for row in rows}) == 20
    assert {row["PRETRAIN_EPISODES"] for row in rows} == {"1000"}
    assert {row["synthetic_unique_episode_count"] for row in rows} == {100}
    assert {row["synthetic_episode_presentations"] for row in rows} == {1000}
    assert bundle["synthetic_unique_episode_count"] == "100"
    masked = [row for row in rows if row["experiment_id"].endswith(
        "no_policy_visible_dependence")]
    assert len(masked) == 10
    assert {row["VINE_FEATURE_MODE"] for row in masked} == {"zero"}
    assert {row["CVAR_OBSERVATION_MODE"] for row in masked} == {"zero"}
    assert {row["CVAR_REWARD_MODE"] for row in masked} == {"full"}


def test_materializer_repeats_exact_paths_without_evaluation_access() -> None:
    source = (ROOT / "rl/materialize_synthetic_presentation_bundle.r").read_text(
        encoding="utf-8")
    assert "rep(seq_along(unique_pretrain), times = repetitions)" in source
    assert "selection_uses_returns_or_diagnostics <- FALSE" in source
    assert "evaluation_data_accessed <- FALSE" in source
    assert "realized_asset_gross" not in source


def test_audit_attests_distinct_diversity_and_presentation_counts() -> None:
    source = (ROOT /
        "publication_pipeline_draft/audit_synthetic_presentation_sweep.py").read_text(
            encoding="utf-8")
    assert '"synthetic_unique_episode_count": 100' in source
    assert '"pretrain_episode_presentations": 1000' in source
    assert "expected_pretrain_actions = 1000" in source


def test_analysis_registers_complete_and_locked_all_scopes() -> None:
    source = (ROOT /
        "publication_pipeline_draft/analyze_synthetic_presentation_response.py").read_text(
            encoding="utf-8")
    assert "complete_only=True" in source
    assert "complete_only=False" in source
    assert '"locked_all_sensitivity_periods": 24' in source
    assert "synthetic_presentation_mechanism_classification.csv" in source
    assert "synthetic_presentation_matched_seed_effects.csv" in source


def test_replay_uses_exact_v2_audit_before_generic_isolated_authorization() -> None:
    source = (ROOT /
        "publication_pipeline_draft/generate_synthetic_presentation_policy_weights.py"
              ).read_text(encoding="utf-8")
    audit_check = source.index('"synthetic_presentation_response_v2"')
    authorization = source.index('"synthetic_dose_checkpoint_audit_v1"')
    assert audit_check < authorization
    assert 'int(audit.get("synthetic_unique_episode_count", -1)) == 100' in source
    assert 'int(audit.get("pretrain_episode_presentations", -1)) == 1000' in source
