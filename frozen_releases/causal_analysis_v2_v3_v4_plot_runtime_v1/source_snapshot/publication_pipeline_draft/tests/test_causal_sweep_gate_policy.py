from __future__ import annotations

import pytest

from publication_pipeline_draft.audit_causal_sweep import (
    AuditError,
    validate_behavior_gate,
)
from publication_pipeline_draft.run_causal_sweep import (
    SweepError,
    select_failed_retry_jobs,
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


def test_retry_selector_uses_exact_failed_keys(tmp_path) -> None:
    jobs = [
        {"experiment_id": "a", "seed": "1"},
        {"experiment_id": "a", "seed": "2"},
        {"experiment_id": "b", "seed": "1"},
    ]
    status = tmp_path / "status.csv"
    status.write_text(
        "experiment_id,seed,passed\n"
        "a,1,true\n"
        "a,2,false\n"
        "b,1,false\n",
        encoding="utf-8",
    )
    selected = select_failed_retry_jobs(jobs, status, 2)
    assert [(row["experiment_id"], row["seed"]) for row in selected] == [
        ("a", "2"), ("b", "1")]


def test_retry_selector_fails_closed_on_wrong_count(tmp_path) -> None:
    jobs = [{"experiment_id": "a", "seed": "1"}]
    status = tmp_path / "status.csv"
    status.write_text("experiment_id,seed,passed\na,1,false\n", encoding="utf-8")
    with pytest.raises(SweepError, match="Expected 2"):
        select_failed_retry_jobs(jobs, status, 2)
