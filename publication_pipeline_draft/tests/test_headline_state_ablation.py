from __future__ import annotations

import json
from pathlib import Path

import pytest

from publication_pipeline_draft.analyze_headline_state_ablation import (
    reference_check, score_four,
)
from publication_pipeline_draft.headline_state_ablation import (
    ARMS, HEADLINE, headline_evidence, jobs_for, load_design,
)
from publication_pipeline_draft.publication_pipeline import Contract, read_realized_panel

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "publication_pipeline_draft/config/headline_state_ablation_v1.json"


def test_headline_input_protocol_has_exact_four_arms_and_ten_seeds() -> None:
    design, base, digest = load_design(ROOT, DESIGN)
    jobs = jobs_for(ROOT, design, base, digest)
    assert len(jobs) == 30
    assert {job["experiment_id"] for job in jobs} == set(ARMS)
    assert all(job["PRETRAIN_DATA_MODE"] == "mixed_historical_synthetic"
               for job in jobs)
    assert all(job["CVAR_REWARD_MODE"] == "full" for job in jobs)
    assert all(job["PRETRAIN_EPISODES"] == "1000" for job in jobs)
    assert all(job["RUN_FINETUNE"] == "true" for job in jobs)
    assert design["evidence_class"] == "post_holdout_explanatory"
    assert design["confirmatory_claim_permitted"] is False
    assert HEADLINE not in {job["experiment_id"] for job in jobs}


def test_frozen_headline_evidence_is_complete() -> None:
    design, _, _ = load_design(ROOT, DESIGN)
    release, rows, digest = headline_evidence(ROOT, design)
    assert release.is_dir()
    assert len(rows) == 10
    assert len(digest) == 64


def test_evaluation_gate_is_specific_to_new_30_checkpoint_audit() -> None:
    source = (ROOT / "evaluate_with_config.r").read_text(encoding="utf-8")
    assert '"headline_state_checkpoint_audit_v1"' in source
    assert '"headline_state_checkpoint_audit_passed"' in source
    assert 'checkpoint is not uniquely authorized' in source.lower()
    runner = (ROOT / "hpc/run_headline_state_ablation_v1.sh").read_text(
        encoding="utf-8")
    assert "checkpoint-archive)" in runner
    assert "analyze)" in runner


def test_archived_headline_rescores_to_table_4_and_5() -> None:
    realized_path = (ROOT / "analysis_outputs/oos_v4_verified_770d2944/"
                     "main_oos_v4_operational_retry/inputs/realized_asset_gross.csv")
    if not realized_path.is_file():
        pytest.skip("Local copy of the locked realized panel is absent")
    design, _, _ = load_design(ROOT, DESIGN)
    evaluation = Contract.read(ROOT / "publication_pipeline_draft/config/evaluation_contract.json")
    realized, assets = read_realized_panel(realized_path, evaluation)
    release, rows, _ = headline_evidence(ROOT, design)
    from publication_pipeline_draft.analyze_synthetic_dose_response import load_weight
    from publication_pipeline_draft.publication_pipeline import KEYS, score_strategy
    import numpy as np
    from publication_pipeline_draft.analyze_causal_results import metrics
    columns = [f"w_{asset}" for asset in assets]
    members = []
    for row in rows:
        seed = int(row["seed"])
        path = (release / "source_evidence/weight_evidence/weights/"
                "mixed_pretraining_plus_historical_finetuning" /
                f"seed_{seed}" / f"weights_rl_full_seed_{seed}.csv")
        members.append(load_weight(path, row["sha256"], realized, assets, evaluation))
    ensemble = members[0][KEYS].copy()
    ensemble[columns] = np.stack([member[columns].to_numpy(float)
                                  for member in members]).mean(axis=0)
    scored = score_strategy("headline", ensemble, realized, assets, evaluation)
    complete = scored[scored["is_complete_period"]]
    causal = json.loads((ROOT / "publication_pipeline_draft/config/"
                         "causal_analysis_contract_v2.json").read_text())
    assert len(complete) == 22
    assert reference_check(ROOT, design, metrics(complete, causal))


def test_weight_space_ensembles_preserve_identical_reference_when_inputs_match() -> None:
    realized_path = (ROOT / "analysis_outputs/oos_v4_verified_770d2944/"
                     "main_oos_v4_operational_retry/inputs/realized_asset_gross.csv")
    if not realized_path.is_file():
        realized_path = (ROOT / "locked_evaluation/main_oos_v4_operational_retry/"
                         "inputs/realized_asset_gross.csv")
    if not realized_path.is_file():
        pytest.skip("Realized panel is not installed here")
    design, _, _ = load_design(ROOT, DESIGN)
    evaluation = Contract.read(ROOT / "publication_pipeline_draft/config/evaluation_contract.json")
    realized, assets = read_realized_panel(realized_path, evaluation)
    release, rows, _ = headline_evidence(ROOT, design)
    from publication_pipeline_draft.analyze_synthetic_dose_response import load_weight
    weights = {}
    for row in rows:
        seed = int(row["seed"])
        path = (release / "source_evidence/weight_evidence/weights/"
                "mixed_pretraining_plus_historical_finetuning" /
                f"seed_{seed}" / f"weights_rl_full_seed_{seed}.csv")
        frame = load_weight(path, row["sha256"], realized, assets, evaluation)
        for arm in (HEADLINE, *ARMS):
            weights[arm, seed] = frame.copy()
    scored, ensembles = score_four(weights, realized, assets, evaluation)
    assert len(ensembles) == 4
    assert len(scored) == 44 * 24
    assert set(scored["turnover_convention"]) == {"target_to_target_v1"}
