from __future__ import annotations

import csv
from pathlib import Path

from publication_pipeline_draft.causal_ablation_protocol import validated_rows, write_matrix


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "publication_pipeline_draft/config/causal_ablation_contract_v2.json"


def test_causal_matrix_has_matched_complete_seed_sets(tmp_path: Path) -> None:
    rows, digest = validated_rows(CONTRACT, tmp_path / "runs")
    experiments = {row["experiment_id"] for row in rows}
    assert len(experiments) == 13
    assert len(rows) == 130
    for experiment in experiments:
        seeds = {row["seed"] for row in rows if row["experiment_id"] == experiment}
        assert len(seeds) == 10
    assert len(digest) == 64


def test_causal_modes_are_separately_identifiable(tmp_path: Path) -> None:
    rows, _ = validated_rows(CONTRACT, tmp_path / "runs")
    by_id = {row["experiment_id"]: row for row in rows}
    vine_only = by_id["zero_vine_features_keep_cvar_observation"]
    cvar_only = by_id["keep_vine_features_zero_cvar_observation"]
    reward = by_id["zero_cvar_reward_keep_state"]
    assert vine_only["VINE_FEATURE_MODE"] == "zero"
    assert vine_only["CVAR_OBSERVATION_MODE"] == "full"
    assert cvar_only["VINE_FEATURE_MODE"] == "full"
    assert cvar_only["CVAR_OBSERVATION_MODE"] == "zero"
    assert reward["CVAR_REWARD_MODE"] == "zero"
    assert reward["VINE_FEATURE_MODE"] == "full"


def test_job_matrix_is_hash_sidecar_protected(tmp_path: Path) -> None:
    rows, digest = validated_rows(CONTRACT, tmp_path / "runs")
    path = tmp_path / "jobs.csv"
    write_matrix(path, rows, digest)
    with path.open(newline="", encoding="utf-8") as stream:
        written = list(csv.DictReader(stream))
    assert len(written) == 130
    assert path.with_suffix(".csv.sha256").is_file()
