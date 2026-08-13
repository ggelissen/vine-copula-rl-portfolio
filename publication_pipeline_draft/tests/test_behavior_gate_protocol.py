from __future__ import annotations

from pathlib import Path

import pytest

from publication_pipeline_draft.behavior_gate_protocol import (
    BehaviorGateProtocolError,
    REPORT_ONLY_TRAINER_FRAGMENTS,
    validate_report_only_trainer,
)


ROOT = Path(__file__).resolve().parents[2]


def test_live_trainer_proves_report_only_gate_wiring() -> None:
    evidence = validate_report_only_trainer(ROOT / "rl/train_rl.r")
    assert evidence["report_only_gate_wiring_valid"] is True
    assert evidence["checked_fragment_count"] == len(REPORT_ONLY_TRAINER_FRAGMENTS)


def test_legacy_strict_only_trainer_is_rejected(tmp_path: Path) -> None:
    trainer = tmp_path / "train_rl.r"
    trainer.write_text(
        "PRETRAIN_BEHAVIOR_GATE_MODE = 'strict'\n"
        "raise RuntimeError('Pre-training behavioural gate failed')\n",
        encoding="utf-8",
    )
    with pytest.raises(BehaviorGateProtocolError, match="report-only gate"):
        validate_report_only_trainer(trainer)

