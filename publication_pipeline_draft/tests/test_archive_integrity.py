from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import Callable

import pytest

import publication_pipeline_draft.locked_evaluation_batch as locked_batch
from publication_pipeline_draft.assemble_publication_provenance import (
    deterministic_tar as provenance_tar,
)
from publication_pipeline_draft.freeze_training_release import (
    deterministic_tar as training_tar,
)
from publication_pipeline_draft.publication_pipeline import ProtocolError
from publication_pipeline_draft.secondary_experiment_protocol import (
    deterministic_tar as secondary_plan_tar,
)


TarWriter = Callable[..., None]


def _make_tree(root: Path, timestamp: int) -> Path:
    source = root / "source_with_unstable_name"
    (source / "nested").mkdir(parents=True)
    (source / "empty").mkdir()
    (source / "alpha.txt").write_text("alpha\n", encoding="utf-8")
    (source / "nested" / "beta.bin").write_bytes(b"\x00\x01beta")
    for path in [source, *source.rglob("*")]:
        os.utime(path, (timestamp, timestamp))
    return source


@pytest.mark.parametrize(
    ("writer", "writes_sidecar"),
    [
        (training_tar, True),
        (secondary_plan_tar, True),
        (provenance_tar, False),
    ],
)
def test_identical_tree_produces_identical_archive_bytes(
    tmp_path: Path, writer: TarWriter, writes_sidecar: bool,
) -> None:
    first_source = _make_tree(tmp_path / "one", 100)
    second_source = _make_tree(tmp_path / "two", 2_000_000_000)
    first_archive = tmp_path / "first-name.tar.gz"
    second_archive = tmp_path / "second-name.tar.gz"

    writer(first_source, first_archive, root_name="stable_release")
    writer(second_source, second_archive, root_name="stable_release")

    first = first_archive.read_bytes()
    second = second_archive.read_bytes()
    assert first == second
    assert first[3] & 0x08 == 0  # no gzip original-filename field
    assert int.from_bytes(first[4:8], "little") == 0
    with tarfile.open(first_archive, "r:gz") as archive:
        assert archive.getnames() == [
            "stable_release",
            "stable_release/alpha.txt",
            "stable_release/empty",
            "stable_release/nested",
            "stable_release/nested/beta.bin",
        ]
        assert all(member.mtime == 0 for member in archive.getmembers())
    assert first_archive.with_suffix(first_archive.suffix + ".sha256").exists() is writes_sidecar


def test_locked_batch_tar_failure_preserves_logs_and_failed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = tmp_path / "evaluation_release"
    config = release / "source_snapshot" / "config" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("seed: 1\n", encoding="utf-8")
    output = tmp_path / "locked_failure"
    bundle = tmp_path / "locked_failure.tar.gz"

    monkeypatch.setattr(
        locked_batch,
        "verify_frozen_sources",
        lambda *_args: {
            "full_training_release": str(tmp_path / "training"),
            "no_vine_training_release": None,
            "evaluation_id": "locked_oos_v1",
            "evaluation_code_contract_sha256": "1" * 64,
            "evaluation_hash_schema_version": 2,
            "evaluation_source_aggregate_sha256": "4" * 64,
            "evaluation_code_sha256": "2" * 64,
            "evaluation_config_sha256": "3" * 64,
            "evaluation_contents_sha256": "5" * 64,
        },
    )
    monkeypatch.setattr(locked_batch, "seed_checkpoints", lambda *_args: [])

    def fake_run_logged(_command, _cwd, _env, logs: Path, label: str) -> float:
        (logs / f"{label}.stdout.txt").write_text("completed\n", encoding="utf-8")
        (logs / f"{label}.stderr.txt").write_text("", encoding="utf-8")
        return 0.01

    def fake_strategy_manifest(path: Path, *_args, **_kwargs) -> None:
        path.write_text("strategy_id\n", encoding="utf-8")

    def fail_tar(*_args, **_kwargs) -> None:
        raise OSError("simulated compressor failure")

    monkeypatch.setattr(locked_batch, "run_logged", fake_run_logged)
    monkeypatch.setattr(locked_batch, "create_strategy_manifest", fake_strategy_manifest)
    monkeypatch.setattr(locked_batch, "deterministic_tar", fail_tar)
    args = argparse.Namespace(
        repo_root=tmp_path,
        evaluation_release=release,
        output=output,
        bundle=bundle,
        rscript="Rscript",
    )

    with pytest.raises(ProtocolError, match="Failure logs were preserved"):
        locked_batch.execute_batch(args)

    manifest = json.loads((output / "locked_batch_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["holdout_accessed"] is True
    assert "simulated compressor failure" in manifest["error"]
    assert (output / "command_logs" / "build_realized_panel.stdout.txt").is_file()
    assert (output / "command_logs" / "benchmarks.stdout.txt").is_file()
    assert (output / "command_logs" / "common_evaluator.stdout.txt").is_file()
    assert not bundle.exists()
    assert not bundle.with_suffix(bundle.suffix + ".sha256").exists()
