from __future__ import annotations

import pytest

from publication_pipeline_draft.causal_ablation_protocol import ENV_FIELDS
from publication_pipeline_draft.merge_causal_operational_retry import (
    OperationalMergeError,
    equivalent_scientific_settings,
)


def jobs() -> tuple[dict[str, str], dict[str, str]]:
    values = {
        "RL_ALGORITHM": "td3", "POLICY_ENCODER": "lstm",
        "VINE_FEATURE_MODE": "full", "CVAR_OBSERVATION_MODE": "full",
        "CVAR_REWARD_MODE": "full", "PRETRAIN_DATA_MODE": "vine_synthetic",
        "RUN_FINETUNE": "true", "SYNTHETIC_RETURNS_FILE": "data/x.RData",
        "CHECKPOINT_PREFIX": "td3_lstm_vine", "LR_ACTOR": "0.00003",
        "LR_CRITIC": "0.0001", "ENTROPY_COEF": "0.005",
    }
    assert set(values) == set(ENV_FIELDS) - {"PRETRAIN_BEHAVIOR_GATE_MODE"}
    common = {"experiment_id": "reference", "seed": "7",
              "job_family": "causal_ablation", **values}
    old = {**common, "output_dir": "data/v2", "contract_sha256": "old"}
    new = {**common, "output_dir": "data/v3", "contract_sha256": "new",
           "PRETRAIN_BEHAVIOR_GATE_MODE": "report_only"}
    return old, new


def test_merge_allows_only_disclosed_gate_and_provenance_changes() -> None:
    old, new = jobs()
    equivalent_scientific_settings(old, new)


def test_merge_rejects_a_changed_scientific_setting() -> None:
    old, new = jobs(); new["LR_ACTOR"] = "0.01"
    with pytest.raises(OperationalMergeError, match="Scientific settings changed"):
        equivalent_scientific_settings(old, new)


def test_merge_requires_report_only_retry_gate() -> None:
    old, new = jobs(); new["PRETRAIN_BEHAVIOR_GATE_MODE"] = "strict"
    with pytest.raises(OperationalMergeError, match="report-only"):
        equivalent_scientific_settings(old, new)
