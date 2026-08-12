from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from freeze_training_release import (  # noqa: E402
    ARCHIVE_TABLES,
    SEED_ARTIFACTS,
    freeze_training_release,
)
from publication_pipeline import ProtocolError  # noqa: E402


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # nosec B324 - fixture


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FreezeTrainingReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.repo = self.base / "repo"
        self.runs = self.repo / "data" / "rl_runs"
        self.runs.mkdir(parents=True)
        (self.repo / "rl").mkdir()
        (self.repo / "rl" / "train_rl.r").write_text("# frozen training code\n", encoding="utf-8")
        (self.repo / "data" / "training_input.bin").write_bytes(b"frozen training data")
        self.seeds = [101, 102]
        status_rows = []
        checkpoint_rows = []
        artifact_rows = []
        policy_rows = []
        gate_rows = []
        sanity_rows = []
        sensitivity_rows = []
        validation_rows = []
        for seed in self.seeds:
            run = self.runs / f"seed_{seed}"
            for relative in SEED_ARTIFACTS:
                path = run / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if relative.endswith("sanity_report.json"):
                    path.write_text(
                        json.dumps({"overall_pass": True, "publication_behavior_pass": True}),
                        encoding="utf-8",
                    )
                elif relative.endswith(".csv"):
                    path.write_text("fixture,value\na,1\n", encoding="utf-8")
                else:
                    path.write_bytes(f"{seed}:{relative}".encode())
            for model in ["pretrained", "full"]:
                checkpoint = run / f"td3_lstm_vine_{model}.pt"
                checkpoint_rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "path": str(checkpoint),
                        "sha256": sha256(checkpoint),
                        "architecture_match": True,
                        "all_checkpoint_tensors_finite": True,
                    }
                )
            for relative in [
                "finetune_validation_metrics.csv",
                "finetune_episode_schedule.csv",
                "finetune_selection.txt",
                "data_hashes.csv",
                "code_hashes.csv",
                "run_manifest.rds",
                "td3_lstm_vine_pretrained.pt",
                "td3_lstm_vine_full.pt",
                "sanity_no_holdout/checkpoint_integrity.csv",
                "sanity_no_holdout/sanity_report.json",
            ]:
                path = run / relative
                artifact_rows.append(
                    {
                        "seed": seed,
                        "artifact": relative,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
            status_rows.append(
                {
                    "seed": seed,
                    "output_dir": str(run),
                    "training_status": 0,
                    "sanity_status": 0,
                    "no_holdout_gate_pass": True,
                    "full_mean_reward": 0.1,
                    "full_mean_terminal_wealth": 120000,
                    "finetune_reward_delta": 0.01,
                    "finetune_terminal_wealth_delta": 1000,
                    "full_median_turnover": 0.4,
                    "full_mean_leverage_gate": 0.8,
                    "full_mean_normalized_entropy": 0.85,
                    "full_mean_effective_positions": 4.0,
                }
            )
            gate_rows.append({"seed": seed, "metric": "finite", "pass": True})
            for model in ["equal_weight", "pretrained", "full"]:
                policy_rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "all_values_finite": True,
                        "hard_constraints_pass": True,
                    }
                )
            sensitivity_rows.append(
                {"seed": seed, "model": "full", "perturbation": "zero_vine"}
            )
            validation_rows.append(
                {"seed": seed, "stage": "finetune_validation", "reward": 0.1}
            )
            sanity_rows.append(
                {
                    "seed": seed,
                    "warning_count": 0,
                    "publication_behavior_pass": True,
                    "overall_pass": True,
                }
            )

        consensus = pd.DataFrame(
            [
                {
                    "artifact_kind": "code",
                    "normalized_path": "rl/train_rl.r",
                    "seed_count": 2,
                    "distinct_hashes": 1,
                    "md5": md5(self.repo / "rl" / "train_rl.r"),
                },
                {
                    "artifact_kind": "data",
                    "normalized_path": "data/training_input.bin",
                    "seed_count": 2,
                    "distinct_hashes": 1,
                    "md5": md5(self.repo / "data" / "training_input.bin"),
                },
            ]
        )
        tables = {
            "status": pd.DataFrame(status_rows),
            "gates": pd.DataFrame(gate_rows),
            "policies": pd.DataFrame(policy_rows),
            "sensitivity": pd.DataFrame(sensitivity_rows),
            "finetune_validation": pd.DataFrame(validation_rows),
            "checkpoints": pd.DataFrame(checkpoint_rows),
            "sanity": pd.DataFrame(sanity_rows),
            "hash_consensus": consensus,
            "artifact_inventory": pd.DataFrame(artifact_rows),
            "episodes": pd.DataFrame(
                [
                    {
                        "seed": seed, "stage": stage, "episode": episode,
                        "reward": 0.1, "terminal_wealth": 110000,
                        "mean_turnover": 0.4, "mean_cvar": 0.07,
                        "mean_gross_exposure": 1.4,
                    }
                    for seed in self.seeds
                    for stage, count in [("pretrain", 2), ("finetune_selection", 1),
                                         ("finetune_refit_all", 1)]
                    for episode in range(1, count + 1)
                ]
            ),
            "updates": pd.DataFrame(
                [
                    {
                        "seed": seed, "stage": stage, "update": update,
                        "critic_loss": 0.01, "actor_loss": 0.02,
                        "twin_q_gap": 0.005, "critic_grad_norm": 0.1,
                        "critic2_grad_norm": 0.1, "actor_grad_norm": 0.05,
                    }
                    for seed in self.seeds
                    for stage, count in [("pretrain", 2), ("finetune_selection", 1),
                                         ("finetune_refit_all", 1)]
                    for update in range(1, count + 1)
                ]
            ),
            "finetune_schedule": pd.DataFrame(
                [
                    {
                        "seed": seed, "stage": stage, "position": position,
                        "original_episode": position,
                    }
                    for seed in self.seeds
                    for stage, count in [("selection_fit", 1), ("all_history_refit", 2)]
                    for position in range(1, count + 1)
                ]
            ),
        }
        archive_tree = self.base / "archive_tree" / "publication_training_artifacts"
        for name, suffix in ARCHIVE_TABLES.items():
            target = archive_tree / suffix
            target.parent.mkdir(parents=True, exist_ok=True)
            tables[name].to_csv(target, index=False)
        self.archive = self.base / "training_artifacts.tar.gz"
        with tarfile.open(self.archive, "w:gz") as handle:
            handle.add(archive_tree, arcname=archive_tree.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_freezes_verified_release_without_holdout(self) -> None:
        output = self.base / "frozen_release"
        bundle = self.base / "frozen_release.tar.gz"
        manifest = freeze_training_release(
            repo_root=self.repo,
            rl_runs=self.runs,
            diagnostics_archive=self.archive,
            output=output,
            expected_seeds=2,
            verify_data=True,
            copy_data=False,
            bundle=bundle,
        )
        self.assertEqual(manifest["release_status"], "frozen_pre_oos")
        self.assertFalse(manifest["holdout_accessed_by_freezer"])
        self.assertTrue((output / "source_snapshot" / "rl" / "train_rl.r").is_file())
        self.assertTrue((output / "seeds" / "seed_101" / "td3_lstm_vine_full.pt").is_file())
        self.assertTrue((output / "CONTENTS.sha256").is_file())
        self.assertTrue(bundle.is_file())
        self.assertTrue(bundle.with_suffix(bundle.suffix + ".sha256").is_file())

    def test_source_hash_mismatch_fails_without_partial_release(self) -> None:
        (self.repo / "rl" / "train_rl.r").write_text("# changed after training\n", encoding="utf-8")
        output = self.base / "must_not_exist"
        with self.assertRaises(ProtocolError):
            freeze_training_release(
                repo_root=self.repo,
                rl_runs=self.runs,
                diagnostics_archive=self.archive,
                output=output,
                expected_seeds=2,
            )
        self.assertFalse(output.exists())

    def test_per_seed_source_snapshot_freezes_exact_training_code(self) -> None:
        frozen_bytes = (self.repo / "rl" / "train_rl.r").read_bytes()
        for seed in self.seeds:
            snapshot = (
                self.runs / f"seed_{seed}" / "source_snapshot" / "rl" /
                "train_rl.r"
            )
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(frozen_bytes)
        (self.repo / "rl" / "train_rl.r").write_text(
            "# later operational code\n", encoding="utf-8"
        )
        output = self.base / "snapshot_backed_release"
        manifest = freeze_training_release(
            repo_root=self.repo,
            rl_runs=self.runs,
            diagnostics_archive=self.archive,
            output=output,
            expected_seeds=2,
        )
        copied = output / "source_snapshot" / "rl" / "train_rl.r"
        self.assertEqual(copied.read_bytes(), frozen_bytes)
        inventory = pd.read_csv(output / "training_snapshot_inventory.csv")
        code = inventory[inventory["normalized_path"] == "rl/train_rl.r"].iloc[0]
        self.assertEqual(code["source_origin"], "per_seed_training_snapshot_consensus")
        self.assertEqual(int(code["training_snapshot_copy_count"]), 2)
        self.assertEqual(manifest["release_status"], "frozen_pre_oos")

    def test_post_holdout_training_release_is_never_labelled_pre_oos(self) -> None:
        output = self.base / "post_holdout_release"
        manifest = freeze_training_release(
            repo_root=self.repo,
            rl_runs=self.runs,
            diagnostics_archive=self.archive,
            output=output,
            expected_seeds=2,
            evidence_class="post_holdout_explanatory",
        )
        self.assertEqual(
            manifest["release_status"],
            "frozen_post_holdout_explanatory_training",
        )
        self.assertEqual(manifest["evidence_class"], "post_holdout_explanatory")
        self.assertFalse(manifest["confirmatory_claims_permitted"])
        read_only = (output / "READ_ONLY_RELEASE.txt").read_text(encoding="utf-8")
        self.assertIn("post-holdout explanatory", read_only)
        self.assertIn("confirmatory claims are forbidden", read_only)
        self.assertNotIn("frozen pre-OOS training release", read_only)


if __name__ == "__main__":
    unittest.main()
