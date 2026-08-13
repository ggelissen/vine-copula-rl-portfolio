from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from publication_pipeline_draft.publication_research_program import (
    ResearchProgramError,
    job_rows,
    validate_program,
)


ROOT = Path(__file__).resolve().parents[2]
PROGRAM = ROOT / "publication_pipeline_draft/config/publication_research_program_v2.json"


def write_program(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "program.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_publication_extension_program_is_complete() -> None:
    validated = validate_program(PROGRAM)
    assert validated.panel_dimensions["global_liquid_etf_18"]["observation_dimension"] == 517
    assert validated.panel_dimensions["liquid_asset_scalability_30_50"]["observation_dimension"] == 466
    universe = json.loads((
        ROOT / "publication_pipeline_draft/config/scalability_universe_v1.json"
    ).read_text())
    assert universe["asset_order"] == json.loads(PROGRAM.read_text())["panels"][1]["asset_order"]
    assert universe["asset_count"] == 40
    rows = job_rows(validated)
    assert len(rows) == 560
    assert {row["algorithm"] for row in rows if row["job_family"] == "rl_algorithm"} == {
        "td3", "ddpg", "sac", "ppo", "a2c"
    }
    assert len({row["seed"] for row in rows}) == 10


def test_retrospective_design_cannot_claim_confirmation(tmp_path: Path) -> None:
    value = json.loads(PROGRAM.read_text())
    value["window_designs"][0]["claim_limit"] = "confirmatory_superiority"
    with pytest.raises(ResearchProgramError, match="cannot make a confirmatory"):
        validate_program(write_program(tmp_path, value))


def test_dense_scalability_state_is_rejected(tmp_path: Path) -> None:
    value = json.loads(PROGRAM.read_text())
    panel = value["panels"][1]
    panel["vine_representation"] = {
        "mode": "dense_all_tree_dvine",
        "maximum_observation_dimension": 800,
    }
    with pytest.raises(ResearchProgramError, match="exceeds its ceiling"):
        validate_program(write_program(tmp_path, value))


def test_partial_seed_success_is_forbidden(tmp_path: Path) -> None:
    value = json.loads(PROGRAM.read_text())
    value["seed_design"]["minimum_successful_seeds"] = 8
    with pytest.raises(ResearchProgramError, match="All preregistered seeds"):
        validate_program(write_program(tmp_path, value))
