from __future__ import annotations

import csv
from pathlib import Path

import pytest

from publication_pipeline_draft.merge_causal_sweep_status import MergeError, merge


def write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def test_exact_disjoint_causal_shards_merge(tmp_path: Path) -> None:
    jobs = [{"experiment_id": f"e{index // 10:02d}", "seed": 1000 + index % 10}
            for index in range(130)]
    statuses = [{**row, "passed": True, "exit_code": 0} for row in jobs]
    job_path = tmp_path / "jobs.csv"; write(job_path, jobs)
    first = tmp_path / "first.csv"; write(first, statuses[:70])
    second = tmp_path / "second.csv"; write(second, statuses[70:])
    output = tmp_path / "merged.csv"
    result = merge(job_path, [first, second], output)
    assert result["job_count"] == result["passed"] == 130
    assert output.is_file()
    assert output.with_suffix(".csv.sha256").is_file()


def test_duplicate_shard_rows_fail_closed(tmp_path: Path) -> None:
    jobs = [{"experiment_id": f"e{index // 10:02d}", "seed": 1000 + index % 10}
            for index in range(130)]
    statuses = [{**row, "passed": True, "exit_code": 0} for row in jobs]
    job_path = tmp_path / "jobs.csv"; write(job_path, jobs)
    first = tmp_path / "first.csv"; write(first, statuses)
    duplicate = tmp_path / "duplicate.csv"; write(duplicate, statuses[:1])
    with pytest.raises(MergeError, match="Duplicate"):
        merge(job_path, [first, duplicate], tmp_path / "merged.csv")
