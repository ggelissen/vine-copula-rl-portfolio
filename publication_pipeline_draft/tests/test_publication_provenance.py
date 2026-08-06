from __future__ import annotations

import csv
import hashlib
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from publication_pipeline_draft.assemble_publication_provenance import (  # noqa: E402
    assemble_publication_provenance,
    deterministic_tar,
    sha256_file,
    write_contents,
)
from publication_pipeline_draft.publication_pipeline import ProtocolError  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sidecar(archive: Path) -> Path:
    path = archive.with_suffix(archive.suffix + ".sha256")
    path.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
    return path


class PublicationProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = tempfile.TemporaryDirectory()
        self.root = Path(self.context.name)
        self.checkpoint = b"checkpoint-bytes"
        self.checkpoint_sha = hashlib.sha256(self.checkpoint).hexdigest()
        self.aggregate_sha = self._make_evaluation_release()
        self._make_training_release()
        self.batch_dir, self.batch_archive, self.batch_sidecar = self._make_batch()
        self.retry_archive, self.retry_sidecar = self._make_retry()
        self.raw = self.root / "prices.csv"
        self.raw.write_text("date,asset\n2026-01-01,1\n", encoding="utf-8")
        self.environment = self.root / "environment.lock"
        self.environment.write_text("python=3.13\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.context.cleanup()

    def _make_evaluation_release(self) -> str:
        release = self.root / "evaluation_release"
        snapshot = release / "source_snapshot"
        sources = {
            "publication_pipeline_draft/publication_pipeline.py": b"print('frozen')\n",
            "publication_pipeline_draft/config/evaluation_contract.json": b"{}\n",
            "publication_pipeline_draft/config/benchmark_contract.json": b"{}\n",
            "config/config.yaml": b"seed: 1\n",
        }
        rows = []
        for relative, value in sources.items():
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
            rows.append(
                {"path": relative, "size_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            )
        write_csv(release / "evaluation_source_inventory.csv", rows)
        payload = "\n".join(f"{row['sha256']}  {row['path']}" for row in rows).encode()
        aggregate = hashlib.sha256(payload).hexdigest()
        (release / "evaluation_release_manifest.json").write_text(
            json.dumps(
                {
                    "release_status": "frozen_pre_holdout_evaluation",
                    "holdout_accessed_by_freezer": False,
                    "evaluation_code_contract_sha256": aggregate,
                }
            ),
            encoding="utf-8",
        )
        write_csv(
            release / "strategy_declaration.csv",
            [
                {
                    "strategy_id": "vine_td3_seed_1",
                    "checkpoint_sha256": self.checkpoint_sha,
                }
            ],
        )
        write_contents(release)
        self.evaluation_release = release
        return aggregate

    def _make_training_release(self) -> None:
        release = self.root / "training_release"
        checkpoint = release / "seeds" / "seed_1" / "td3_lstm_vine_full.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(self.checkpoint)
        source = release / "source_snapshot" / "rl" / "train_rl.r"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# frozen training code\n", encoding="utf-8")
        data_hash = hashlib.md5(b"data").hexdigest()  # nosec - recorded compatibility hash
        write_csv(
            release / "training_snapshot_inventory.csv",
            [
                {
                    "artifact_kind": "code",
                    "normalized_path": "rl/train_rl.r",
                    "expected_md5": hashlib.md5(source.read_bytes()).hexdigest(),  # nosec
                    "sha256": sha256_file(source),
                },
                {
                    "artifact_kind": "data",
                    "normalized_path": "data/prices.csv",
                    "expected_md5": data_hash,
                    "sha256": "",
                },
            ],
        )
        (release / "training_release_manifest.json").write_text(
            json.dumps(
                {
                    "release_status": "frozen_pre_oos",
                    "holdout_accessed_by_freezer": False,
                }
            ),
            encoding="utf-8",
        )
        write_contents(release)
        self.training_release = release

    def _make_batch(self) -> tuple[Path, Path, Path]:
        batch = self.root / "successful_batch"
        batch.mkdir()
        (batch / "locked_batch_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "holdout_accessed": True,
                    "evaluation_release_sha256": self.aggregate_sha,
                }
            ),
            encoding="utf-8",
        )
        write_csv(
            batch / "strategy_manifest.csv",
            [
                {
                    "strategy_id": "vine_td3_seed_1",
                    "checkpoint_sha256": self.checkpoint_sha,
                }
            ],
        )
        (batch / "result.txt").write_text("immutable result\n", encoding="utf-8")
        archive = self.root / "successful_batch.tar.gz"
        deterministic_tar(batch, archive)
        return batch, archive, sidecar(archive)

    def _make_retry(self) -> tuple[Path, Path]:
        retry = self.root / "failed_retry"
        (retry / "command_logs").mkdir(parents=True)
        (retry / "inputs").mkdir()
        (retry / "locked_batch_manifest.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "holdout_accessed": True,
                    "error_type": "ProtocolError",
                    "error": "Locked command benchmarks failed with exit code 1",
                }
            ),
            encoding="utf-8",
        )
        (retry / "command_logs" / "benchmarks.stderr.txt").write_text(
            "mechanical failure\n", encoding="utf-8"
        )
        (retry / "inputs" / "realized_asset_gross.csv").write_text(
            "decision_date,g_A\n2026-01-01,1\n", encoding="utf-8"
        )
        archive = self.root / "failed_retry.tar.gz"
        deterministic_tar(retry, archive)
        return archive, sidecar(archive)

    def _assemble(self, **updates: object) -> tuple[Path, Path, dict[str, object]]:
        output = self.root / "provenance"
        bundle = self.root / "provenance.tar.gz"
        arguments: dict[str, object] = {
            "successful_batch_directory": self.batch_dir,
            "successful_batch_archive": self.batch_archive,
            "successful_batch_sidecar": self.batch_sidecar,
            "evaluation_release": self.evaluation_release,
            "training_release": self.training_release,
            "failed_retries": [(self.retry_archive, self.retry_sidecar)],
            "no_failed_retries": False,
            "raw_data": [("prices", self.raw)],
            "external_raw_data": [],
            "environment_manifests": [self.environment],
            "output": output,
            "bundle": bundle,
            "max_copy_bytes": 1,
        }
        arguments.update(updates)
        manifest = assemble_publication_provenance(**arguments)
        return output, bundle, manifest

    def test_builds_verified_incident_and_hash_package(self) -> None:
        output, bundle, manifest = self._assemble()
        self.assertEqual(manifest["release_status"], "publication_provenance_evidence")
        self.assertTrue((output / "CONTENTS.sha256").is_file())
        self.assertTrue((output / "validation_report.json").is_file())
        self.assertTrue((output / "incident_timeline.csv").is_file())
        self.assertTrue(bundle.is_file())
        self.assertTrue(bundle.with_suffix(bundle.suffix + ".sha256").is_file())
        report = json.loads((output / "validation_report.json").read_text())
        self.assertEqual(report["required_check_failures"], 0)
        roles = {row["hash_role"] for row in read_rows(output / "hash_registry.csv")}
        self.assertTrue(
            {
                "successful_locked_batch_archive", "evaluation_release_aggregate",
                "evaluation_config", "benchmark_config", "training_code",
                "training_data", "training_checkpoint", "raw_market_data",
                "software_environment", "failed_retry_archive",
            }.issubset(roles)
        )
        inventory = read_rows(output / "inventory.csv")
        raw = next(row for row in inventory if row["logical_name"] == "prices")
        self.assertEqual(raw["copied"], "False")
        checkpoint = next(
            row for row in inventory if row["logical_name"].endswith("td3_lstm_vine_full.pt")
        )
        self.assertEqual(checkpoint["copied"], "False")
        incident = read_rows(output / "incident_timeline.csv")[0]
        self.assertEqual(incident["failure_stage"], "benchmarks")
        self.assertEqual(incident["contains_realized_panel"], "True")

    def test_external_licensed_raw_data_is_explicitly_accepted(self) -> None:
        digest = "a" * 64
        output, _, _ = self._assemble(
            raw_data=[],
            external_raw_data=[("licensed_prices", digest, "doi:10.1/example", "Vendor license")],
        )
        inventory = read_rows(output / "inventory.csv")
        external = next(row for row in inventory if row["logical_name"] == "licensed_prices")
        self.assertEqual(external["external"], "True")
        self.assertEqual(external["sha256"], digest)

    def test_archive_hash_mismatch_fails_without_output(self) -> None:
        self.batch_sidecar.write_text(f"{'0' * 64}  {self.batch_archive.name}\n")
        output = self.root / "provenance"
        with self.assertRaises(ProtocolError):
            self._assemble()
        self.assertFalse(output.exists())

    def test_missing_raw_provenance_fails_closed(self) -> None:
        with self.assertRaises(ProtocolError):
            self._assemble(raw_data=[], external_raw_data=[])


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


if __name__ == "__main__":
    unittest.main()
