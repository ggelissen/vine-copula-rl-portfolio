from __future__ import annotations

from publication_pipeline_draft.behavior_gate_protocol import (
    validate_report_only_trainer,
)
from publication_pipeline_draft.merge_causal_three_revision_retry import (
    COUNTS,
    V3_STRICT_ONLY_TRAINER_SHA256,
    V4_REPORT_ONLY_TRAINER_SHA256,
)
from publication_pipeline_draft.freeze_causal_analysis_plan import SOURCES


def test_three_revision_recovery_has_exact_disclosed_counts() -> None:
    assert COUNTS == {"v2": 70, "v3": 31, "v4": 29}
    assert sum(COUNTS.values()) == 130


def test_trainer_hashes_are_distinct_and_well_formed() -> None:
    assert len(V3_STRICT_ONLY_TRAINER_SHA256) == 64
    assert len(V4_REPORT_ONLY_TRAINER_SHA256) == 64
    assert V3_STRICT_ONLY_TRAINER_SHA256 != V4_REPORT_ONLY_TRAINER_SHA256


def test_live_v4_trainer_implements_report_only(root_dir=None) -> None:
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    assert validate_report_only_trainer(root / "rl/train_rl.r")[
        "report_only_gate_wiring_valid"] is True


def test_analysis_freeze_includes_three_revision_evidence() -> None:
    assert "publication_pipeline_draft/merge_causal_three_revision_retry.py" in SOURCES
    assert "publication_pipeline_draft/PUBLICATION_EXTENSION_V4_GATE_RECOVERY.md" in SOURCES
    assert "publication_pipeline_draft/V4_RETRY29_EVIDENCE.md" in SOURCES
