from __future__ import annotations

import pytest

from publication_pipeline_draft.audit_causal_sweep import (
    AuditError,
    validate_behavior_gate,
)


def row(metric: str, value: float, passed: bool) -> dict[str, str]:
    return {"metric": metric, "value": str(value), "pass": str(passed)}


def test_report_only_retains_finite_economic_behavior_failures() -> None:
    gate = [row("mean_turnover", 1.3, False),
            row("max_position_limit_violation", 0.0, True)]
    assert validate_behavior_gate(gate, "report_only", "test") == ["mean_turnover"]


def test_strict_mode_rejects_the_same_economic_failure() -> None:
    with pytest.raises(AuditError, match="Strict behavioral gate failed"):
        validate_behavior_gate([row("mean_turnover", 1.3, False)],
                               "strict", "test")


@pytest.mark.parametrize("metric", [
    "gate_gross_mae", "max_position_limit_violation",
])
def test_report_only_never_waives_hard_constraint_failures(metric: str) -> None:
    with pytest.raises(AuditError, match="Hard-constraint gate failed"):
        validate_behavior_gate([row(metric, 0.1, False)],
                               "report_only", "test")


def test_report_only_never_waives_nonfinite_diagnostics() -> None:
    with pytest.raises(AuditError, match="non-finite"):
        validate_behavior_gate([row("mean_turnover", float("nan"), False)],
                               "report_only", "test")
