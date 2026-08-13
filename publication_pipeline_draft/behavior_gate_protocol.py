#!/usr/bin/env python3
"""Static fail-closed checks for the report-only causal training gate."""

from __future__ import annotations

from pathlib import Path


class BehaviorGateProtocolError(RuntimeError):
    pass


REPORT_ONLY_TRAINER_FRAGMENTS = (
    "Frozen causal gate wiring: report_only_v4_20260812",
    'pretrain_behavior_gate_mode <- tolower(Sys.getenv(',
    '"PRETRAIN_BEHAVIOR_GATE_MODE", "strict"',
    'c("strict", "report_only")',
    "PRETRAIN_BEHAVIOR_GATE_MODE = os.environ.get(",
    "'PRETRAIN_BEHAVIOR_GATE_MODE', 'strict').lower()",
    "'pretrain_behavior_gate_mode': PRETRAIN_BEHAVIOR_GATE_MODE",
    "structural_metrics = {'gate_gross_mae', 'max_position_limit_violation'}",
    "if PRETRAIN_BEHAVIOR_GATE_MODE == 'strict'",
    "Continuing under the frozen report_only causal-control protocol.",
)


def validate_report_only_trainer(path: Path) -> dict[str, object]:
    """Prove that a trainer records and enforces report-only semantics.

    This is intentionally a static contract check.  It prevents a release from
    freezing a trainer that receives PRETRAIN_BEHAVIOR_GATE_MODE in its job
    matrix but silently falls back to the historical strict gate.
    """
    if not path.is_file():
        raise BehaviorGateProtocolError(f"Trainer source was not found: {path}")
    text = path.read_text(encoding="utf-8")
    missing = [fragment for fragment in REPORT_ONLY_TRAINER_FRAGMENTS
               if fragment not in text]
    if missing:
        labels = ", ".join(repr(fragment) for fragment in missing)
        raise BehaviorGateProtocolError(
            f"Trainer does not implement the frozen report-only gate: {path}; "
            f"missing={labels}")
    return {"trainer": str(path), "report_only_gate_wiring_valid": True,
            "checked_fragment_count": len(REPORT_ONLY_TRAINER_FRAGMENTS)}
