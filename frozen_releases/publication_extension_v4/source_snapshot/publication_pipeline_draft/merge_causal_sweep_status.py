#!/usr/bin/env python3
"""Merge disjoint causal-sweep shards into the exact 130-job status table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class MergeError(RuntimeError):
    pass


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise MergeError(f"Status/job file not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(jobs_path: Path, statuses: list[Path], output: Path) -> dict[str, Any]:
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise MergeError(f"Merged status already exists: {output}")
    _, jobs = read(jobs_path)
    expected = {(row["experiment_id"], row["seed"]) for row in jobs}
    if len(jobs) != len(expected) or len(jobs) != 130:
        raise MergeError("Job matrix is not the exact 130-job causal contract.")
    fields: list[str] | None = None
    merged: dict[tuple[str, str], dict[str, str]] = {}
    input_hashes = []
    for path in statuses:
        current_fields, rows = read(path)
        if fields is None:
            fields = current_fields
        if current_fields != fields:
            raise MergeError(f"Status schemas differ: {path}")
        for row in rows:
            key = row.get("experiment_id", ""), row.get("seed", "")
            if key not in expected:
                raise MergeError(f"Undeclared status row in {path}: {key}")
            if key in merged:
                raise MergeError(f"Duplicate status row across shards: {key}")
            merged[key] = row
        input_hashes.append({"path": str(path.resolve()), "sha256": sha256(path),
                             "rows": len(rows)})
    if set(merged) != expected:
        missing = sorted(expected - set(merged))
        raise MergeError(f"Causal status is incomplete; missing {len(missing)} jobs.")
    assert fields is not None
    rows = [merged[key] for key in sorted(merged)]
    passed = sum(row.get("passed", "").lower() in {"true", "1"} for row in rows)
    if passed != 130:
        raise MergeError(f"Only {passed}/130 jobs passed; failed rows remain evidence.")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{sha256(output)}  {output.name}\n", encoding="ascii")
    result = {
        "schema_version": 1, "status": "complete_causal_status_merged",
        "job_count": len(rows), "passed": passed, "shard_count": len(statuses),
        "jobs_sha256": sha256(jobs_path), "merged_status_sha256": sha256(output),
        "inputs": input_hashes, "seed_substitution_permitted": False,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--statuses", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = merge(args.jobs.resolve(),
                       [path.resolve() for path in args.statuses], args.output)
    except (OSError, ValueError, MergeError) as error:
        print(f"CAUSAL STATUS MERGE FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
