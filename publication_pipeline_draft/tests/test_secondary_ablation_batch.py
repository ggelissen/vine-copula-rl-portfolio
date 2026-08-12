from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from publication_pipeline_draft.freeze_training_release import deterministic_tar
from publication_pipeline_draft.publication_pipeline import (
    Contract,
    score_strategy,
)
from publication_pipeline_draft.secondary_ablation_batch import (
    EVIDENCE_CLASS,
    INTERPRETATION,
    SecondaryAblationError,
    execute_secondary_ablation,
    mean_ensemble,
    sha256_file,
    validate_contract,
    verify_archived_scores,
)


ROOT = Path(__file__).resolve().parents[2]
LIVE_CONTRACT = ROOT / "publication_pipeline_draft/config/secondary_evaluation_contract_v1.json"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_contents(root: Path) -> None:
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name != "CONTENTS.sha256"
    ]
    (root / "CONTENTS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


class SyntheticRunner:
    def __init__(
        self,
        realized: pd.DataFrame,
        malformed: bool = False,
        replay_drift: bool = False,
    ) -> None:
        self.realized = realized
        self.malformed = malformed
        self.replay_drift = replay_drift
        self.calls: list[dict[str, str]] = []

    def __call__(
        self,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        logs: Path,
        label: str,
    ) -> float:
        del command, cwd
        self.calls.append(
            {
                "label": label,
                "model": env["EVAL_CHECKPOINT_MODELS"],
                "mode": env["VINE_OBSERVATION_MODE"],
                "weights_only": env["EVAL_WEIGHTS_ONLY"],
            }
        )
        seed_name = Path(env["EVAL_MODEL_DIR"]).name
        seed = int(seed_name.removeprefix("seed_"))
        model = env["EVAL_CHECKPOINT_MODELS"]
        output = Path(env["EVAL_OUTPUT_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        weights = self.realized[["window_id", "decision_date", "holding_end_date"]].copy()
        if label.startswith("full_v4_inference_replay"):
            first = 0.54 if seed % 2 else 0.46
            if self.replay_drift:
                first += 0.01
        elif model == "pretrained":
            first = 0.50 + (seed % 2) * 0.04
        else:
            first = 0.44 + (seed % 2) * 0.08
        weights["w_A"] = first
        weights["w_B"] = 1.0 - first
        if self.malformed and len(self.calls) == 1:
            weights = weights.iloc[:-1].copy()
        weights.to_csv(output / f"weights_rl_{model}_{seed_name}.csv", index=False)
        (logs / f"{label}.stdout.txt").write_text("synthetic success\n", encoding="utf-8")
        (logs / f"{label}.stderr.txt").write_text("", encoding="utf-8")
        return 0.01


class SecondaryAblationBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = tempfile.TemporaryDirectory()
        self.base = Path(self.context.name)
        self.repo = self.base / "repo"
        (self.repo / "rl").mkdir(parents=True)
        (self.repo / "config").mkdir(parents=True)
        (self.repo / "evaluate_with_config.r").write_text("# wrapper\n", encoding="utf-8")
        (self.repo / "rl" / "evaluate_rl.r").write_text(
            "# supports EVAL_CHECKPOINT_MODELS\n", encoding="utf-8"
        )
        self.runtime_config = self.repo / "config" / "config.yaml"
        self.runtime_config.write_text("seed: 1\n", encoding="utf-8")
        self.evaluation_contract = self.base / "evaluation_contract.json"
        self.evaluation_contract.write_text(
            json.dumps(self._evaluation_contract(), indent=2) + "\n", encoding="utf-8"
        )
        self.realized = pd.DataFrame(
            {
                "window_id": ["locked_oos_v1"] * 3,
                "decision_date": ["2024-01-01", "2024-02-01", "2024-03-01"],
                "holding_end_date": ["2024-01-31", "2024-02-29", "2024-03-31"],
                "trading_days": [21, 20, 21],
                "is_complete_period": [True, True, True],
                "g_A": [1.02, 0.99, 1.03],
                "g_B": [1.01, 1.02, 0.98],
            }
        )
        self.full_seeds = [1, 2]
        self.no_vine_seeds = [11, 12]
        self.full_release = self._make_release("full_release", self.full_seeds, "full", True)
        self.no_vine_release = self._make_release(
            "no_vine_release", self.no_vine_seeds, "zero", False
        )
        self.archive, self.archive_sidecar = self._make_successful_archive()
        self.contract = self._write_secondary_contract()

    def tearDown(self) -> None:
        self.context.cleanup()

    @staticmethod
    def _evaluation_contract() -> dict[str, object]:
        return {
            "schema_version": 1,
            "evaluation_id": "locked_oos_v1",
            "expected_locked_periods_per_window": 3,
            "minimum_complete_periods_per_window": 2,
            "primary_sample_scope": "complete_periods",
            "periods_per_year": 12,
            "initial_wealth": 100000.0,
            "net_exposure": 1.0,
            "gross_leverage": 1.5,
            "max_long_weight": 0.6,
            "max_short_weight": 0.2,
            "turnover_cost": 0.001,
            "annual_short_borrow_rate": 0.03,
            "annual_cash_borrow_rate": 0.02,
            "annual_risk_free_rate": 0.0,
            "crra_gamma": 2.0,
            "primary_benchmark_id": "equal_weight",
            "primary_strategy_id": "full_ensemble",
            "primary_superiority_test": "one_sided_paired_moving_block_bootstrap_crra",
            "primary_superiority_alpha": 0.05,
            "secondary_multiplicity_control": "holm_within_primary_vs_alternative_family",
            "bootstrap_replications": 999,
            "bootstrap_block_length": 2,
            "inference_seed": 1,
            "weight_tolerance": 0.000001,
            "require_weight_log_hashes": True,
            "require_checkpoint_hash_for_trained_models": True,
            "require_code_and_config_hashes": True,
            "predeclared_ensembles": [],
        }

    def _make_release(
        self, name: str, seeds: list[int], mode: str, include_pretrained: bool
    ) -> Path:
        release = self.base / name
        for seed in seeds:
            directory = release / "seeds" / f"seed_{seed}"
            directory.mkdir(parents=True)
            (directory / "td3_lstm_vine_full.pt").write_bytes(f"full-{seed}".encode())
            (directory / "td3_lstm_vine_pretrained.pt").write_bytes(
                f"pretrained-{seed}".encode()
            )
            (directory / "vine_observation_mode.txt").write_text(mode + "\n", encoding="utf-8")
            sanity = directory / "sanity_no_holdout"
            sanity.mkdir()
            pd.DataFrame(
                [
                    {
                        "model": "pretrained", "architecture_match": True,
                        "all_checkpoint_tensors_finite": True,
                        "actor_parameters": 1234, "update_count": 100,
                    },
                    {
                        "model": "full", "architecture_match": True,
                        "all_checkpoint_tensors_finite": True,
                        "actor_parameters": 1234, "update_count": 150 + seed % 2,
                    },
                ]
            ).to_csv(sanity / "checkpoint_integrity.csv", index=False)
            (sanity / "sanity_report.json").write_text(
                json.dumps(
                    {
                        "vine_observation_mode": mode,
                        "overall_pass": True,
                        "publication_behavior_pass": True,
                        "obs_dim": 88,
                        "action_dim": 7,
                        "vine_dim": 21,
                    }
                ),
                encoding="utf-8",
            )
        code_sources = {
            "config/config.yaml": (
                "vine:\n  sim_cores: " + ("2" if mode == "zero" else "1") +
                "\nablation:\n  zero_vine_state: " +
                ("true" if mode == "zero" else "false") + "\nagent:\n  hidden: 64\n"
            ),
            "rl/action_projection.py": "IDENTICAL_ACTION_PROJECTION = True\n",
            "run_with_config.r": (
                'set_default_env("VINE_OBSERVATION_MODE", "zero")\n'
                if mode == "zero" else "# legacy full launcher\n"
            ),
            "rl/train_rl.r": (
                'vine_observation_mode <- Sys.getenv("VINE_OBSERVATION_MODE", "full")\n'
                "make_env(vine_observation_mode = vine_observation_mode)\n"
                if mode == "zero" else "# legacy full trainer\n"
            ),
            "rl/rl_environment.r": (
                "vine_observation <- if (no_vine_observation) numeric(63) else vine_state\n"
                "cvar_observation <- if (no_vine_observation) 0 else last_cvar\n"
                if mode == "zero" else "# legacy full environment\n"
            ),
            "helper/reproducibility.r": (
                'no_vine_signal_mask <- "explicit_vine_and_scenario_cvar_v1"\n'
                'Sys.getenv("VINE_OBSERVATION_MODE")\n'
                if mode == "zero" else "# legacy full manifest\n"
            ),
            "rl/training_sanity_check.r": (
                "cvar_observation <- if (no_vine_observation) 0 else last_cvar\n"
                'no_vine_signal_mask <- "explicit_vine_and_scenario_cvar_v1"\n'
                if mode == "zero" else "# legacy full sanity\n"
            ),
        }
        for relative, content in code_sources.items():
            target = release / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        pd.DataFrame(
            [
                {
                    "artifact_kind": "code", "normalized_path": path,
                    "expected_md5": f"hash-{index}",
                }
                for index, path in enumerate(code_sources)
            ]
        ).to_csv(release / "training_snapshot_inventory.csv", index=False)
        post_holdout = mode == "zero"
        (release / "training_release_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "release_status": (
                        "frozen_post_holdout_explanatory_training"
                        if post_holdout else "frozen_pre_oos"
                    ),
                    "evidence_class": (
                        "post_holdout_explanatory" if post_holdout else "pre_oos"
                    ),
                    "confirmatory_claims_permitted": not post_holdout,
                    "holdout_accessed_by_freezer": False,
                    "seed_artifact_count": len(seeds),
                }
            ),
            encoding="utf-8",
        )
        write_contents(release)
        return release

    def _weights(self, first: float) -> pd.DataFrame:
        frame = self.realized[["window_id", "decision_date", "holding_end_date"]].copy()
        frame["w_A"] = first
        frame["w_B"] = 1.0 - first
        return frame

    def _make_successful_archive(self) -> tuple[Path, Path]:
        tree = self.base / "successful_v4"
        (tree / "inputs").mkdir(parents=True)
        (tree / "weights").mkdir()
        (tree / "benchmark_weights").mkdir()
        (tree / "publication_results" / "raw").mkdir(parents=True)
        realized_path = tree / "inputs" / "realized_asset_gross.csv"
        self.realized.to_csv(realized_path, index=False)
        benchmark = self._weights(0.55)
        benchmark_path = tree / "benchmark_weights" / "weights_benchmark_a.csv"
        benchmark.to_csv(benchmark_path, index=False)

        manifest_rows: list[dict[str, object]] = [
            {
                "strategy_id": "equal_weight",
                "seed": "",
                "role": "benchmark",
                "weight_log_path": "GENERATE_EQUAL_WEIGHT",
                "weight_log_sha256": "",
                "checkpoint_sha256": "not_applicable",
            },
            {
                "strategy_id": "benchmark_a",
                "seed": "",
                "role": "benchmark",
                "weight_log_path": "benchmark_weights/weights_benchmark_a.csv",
                "weight_log_sha256": sha256_file(benchmark_path),
                "checkpoint_sha256": "not_applicable",
            },
        ]
        full_frames: dict[int, pd.DataFrame] = {}
        for seed, first in [(1, 0.54), (2, 0.46)]:
            frame = self._weights(first)
            path = tree / "weights" / f"weights_rl_full_seed_{seed}.csv"
            frame.to_csv(path, index=False)
            checkpoint = self.full_release / "seeds" / f"seed_{seed}" / "td3_lstm_vine_full.pt"
            manifest_rows.append(
                {
                    "strategy_id": f"vine_td3_seed_{seed}",
                    "seed": seed,
                    "role": "proposed",
                    "weight_log_path": f"weights/{path.name}",
                    "weight_log_sha256": sha256_file(path),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            )
            full_frames[seed] = frame
        manifest_path = tree / "strategy_manifest.csv"
        write_csv(
            manifest_path,
            manifest_rows,
            [
                "strategy_id", "seed", "role", "weight_log_path",
                "weight_log_sha256", "checkpoint_sha256",
            ],
        )

        contract = Contract.read(self.evaluation_contract)
        realized_for_score = self.realized.copy()
        for column in ["decision_date", "holding_end_date"]:
            realized_for_score[column] = pd.to_datetime(realized_for_score[column])
        score_weights = {
            "equal_weight": self._weights(0.5),
            "benchmark_a": benchmark,
            "full_ensemble": mean_ensemble(full_frames, "full_ensemble", ["A", "B"], contract),
        }
        for frame in score_weights.values():
            for column in ["decision_date", "holding_end_date"]:
                frame[column] = pd.to_datetime(frame[column])
        scored = pd.concat(
            [
                score_strategy(strategy_id, frame, realized_for_score, ["A", "B"], contract)
                for strategy_id, frame in score_weights.items()
            ],
            ignore_index=True,
        )
        scored.to_csv(
            tree / "publication_results" / "raw" / "scored_monthly_panel.csv",
            index=False,
        )

        artifact_rows = [
            {"artifact": "evaluation_contract", "sha256": sha256_file(self.evaluation_contract)},
            {"artifact": "realized_panel", "sha256": sha256_file(realized_path)},
            {"artifact": "strategy_manifest", "sha256": sha256_file(manifest_path)},
            {"artifact": "weights:equal_weight", "sha256": "generated"},
            {"artifact": "weights:benchmark_a", "sha256": sha256_file(benchmark_path)},
        ]
        for seed in self.full_seeds:
            path = tree / "weights" / f"weights_rl_full_seed_{seed}.csv"
            artifact_rows.append(
                {"artifact": f"weights:vine_td3_seed_{seed}", "sha256": sha256_file(path)}
            )
        write_csv(
            tree / "publication_results" / "raw" / "input_hashes.csv",
            artifact_rows,
            ["artifact", "sha256"],
        )
        (tree / "publication_results" / "run_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "evaluation_id": "locked_oos_v1",
                    "contract_sha256": sha256_file(self.evaluation_contract),
                    "realized_panel_sha256": sha256_file(realized_path),
                    "strategy_manifest_sha256": sha256_file(manifest_path),
                }
            ),
            encoding="utf-8",
        )
        (tree / "locked_batch_manifest.json").write_text(
            json.dumps({"schema_version": 1, "status": "complete", "holdout_accessed": True}),
            encoding="utf-8",
        )
        archive = self.base / "successful_v4.tar.gz"
        deterministic_tar(tree, archive)
        return archive, archive.with_suffix(archive.suffix + ".sha256")

    def _write_secondary_contract(self) -> Path:
        economics = {
            key: self._evaluation_contract()[key]
            for key in [
                "periods_per_year", "initial_wealth", "net_exposure", "gross_leverage",
                "max_long_weight", "max_short_weight", "turnover_cost",
                "annual_short_borrow_rate", "annual_cash_borrow_rate",
                "annual_risk_free_rate", "crra_gamma",
            ]
        }
        contract = {
            "schema_version": 1,
            "protocol_id": "synthetic_post_holdout_explanatory",
            "evidence_class": EVIDENCE_CLASS,
            "same_sample_interpretation": INTERPRETATION,
            "confirmatory_claims_permitted": False,
            "successful_v4": {
                "archive_sha256": sha256_file(self.archive),
                "evaluation_id": "locked_oos_v1",
                "required_artifacts": {
                    "batch_manifest": "locked_batch_manifest.json",
                    "realized_panel": "inputs/realized_asset_gross.csv",
                    "strategy_manifest": "strategy_manifest.csv",
                    "evaluation_run_manifest": "publication_results/run_manifest.json",
                    "input_hashes": "publication_results/raw/input_hashes.csv",
                    "scored_monthly_panel": "publication_results/raw/scored_monthly_panel.csv",
                },
            },
            "evaluation": {
                "economics_fields": list(economics),
                "expected_economics": economics,
                "benchmark_strategy_ids": ["equal_weight", "benchmark_a"],
                "archive_score_tolerance": 1e-12,
                "inference_replay_weight_tolerance": 1e-10,
                "weight_tolerance": 1e-6,
            },
            "matched_design": {
                "required_equal_code_paths": [
                    "rl/action_projection.py",
                ],
                "allowed_config_difference_paths": [
                    "vine.sim_cores", "ablation.zero_vine_state",
                ],
                "required_no_vine_source_markers": {
                    "run_with_config.r": ['set_default_env("VINE_OBSERVATION_MODE"'],
                    "rl/train_rl.r": [
                        'vine_observation_mode <- Sys.getenv("VINE_OBSERVATION_MODE", "full")',
                        "vine_observation_mode = vine_observation_mode",
                    ],
                    "rl/rl_environment.r": [
                        "vine_observation <- if (no_vine_observation)",
                        "cvar_observation <- if (no_vine_observation) 0",
                    ],
                    "helper/reproducibility.r": [
                        "no_vine_signal_mask", '"VINE_OBSERVATION_MODE"',
                    ],
                    "rl/training_sanity_check.r": [
                        "cvar_observation <- if (no_vine_observation) 0",
                        "explicit_vine_and_scenario_cvar_v1",
                    ],
                },
                "require_equal_actor_parameter_count": True,
                "require_equal_observation_and_action_dimensions": True,
                "require_equal_pretraining_update_count": True,
                "full_update_count_rule": "at_least_pretraining_update_count",
            },
            "experiments": {
                "full_reference": {
                    "evidence_class": EVIDENCE_CLASS,
                    "source": "successful_v4_archive",
                    "checkpoint_model": "full",
                    "observation_mode": "full",
                    "allow_legacy_missing_mode": True,
                    "expected_seeds": self.full_seeds,
                    "ensemble_strategy_id": "full_ensemble",
                },
                "no_vine": {
                    "evidence_class": EVIDENCE_CLASS,
                    "source": "no_vine_training_release",
                    "checkpoint_model": "full",
                    "observation_mode": "zero",
                    "ablation_scope": "policy_visible_vine_state_only_reward_cvar_retained",
                    "expected_seeds": self.no_vine_seeds,
                    "ensemble_strategy_id": "no_vine_ensemble",
                },
                "pretrained_only": {
                    "evidence_class": EVIDENCE_CLASS,
                    "source": "full_training_release",
                    "checkpoint_model": "pretrained",
                    "observation_mode": "full",
                    "allow_legacy_missing_mode": True,
                    "expected_seeds": self.full_seeds,
                    "ensemble_strategy_id": "pretrained_ensemble",
                },
            },
            "outputs": {
                "release_status": "frozen_post_holdout_explanatory_ablation",
                "same_sample_tests": "descriptive_only_no_confirmatory_tests",
            },
        }
        path = self.base / "secondary_contract.json"
        path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        return path

    def _execute(self, runner: SyntheticRunner):
        output = self.base / "post_holdout_explanatory_secondary_release"
        bundle = self.base / "post_holdout_explanatory_secondary_release.tar.gz"
        manifest = execute_secondary_ablation(
            repo_root=self.repo,
            contract_path=self.contract,
            successful_archive=self.archive,
            successful_sidecar=self.archive_sidecar,
            evaluation_contract_path=self.evaluation_contract,
            runtime_config=self.runtime_config,
            full_training_release=self.full_release,
            no_vine_training_release=self.no_vine_release,
            output=output,
            bundle=bundle,
            runner=runner,
        )
        return output, bundle, manifest

    def test_live_contract_is_permanently_explanatory(self) -> None:
        contract = validate_contract(json.loads(LIVE_CONTRACT.read_text(encoding="utf-8")))
        self.assertEqual(contract["evidence_class"], EVIDENCE_CLASS)
        self.assertFalse(contract["confirmatory_claims_permitted"])
        self.assertEqual(
            contract["successful_v4"]["archive_sha256"],
            "770d2944f915d0ad21ae9af32e31d68d652fdb54e98939caeab45c327b4e5ea1",
        )

    def test_builds_explanatory_release_and_uses_explicit_checkpoint_models(self) -> None:
        runner = SyntheticRunner(self.realized)
        output, bundle, manifest = self._execute(runner)
        self.assertEqual(manifest["evidence_class"], EVIDENCE_CLASS)
        self.assertFalse(manifest["confirmatory_claims_permitted"])
        self.assertTrue(manifest["economic_replay_verified"])
        self.assertTrue(manifest["full_inference_replay_verified"])
        self.assertTrue(manifest["matched_design_verified"])
        self.assertTrue(bundle.is_file())
        self.assertTrue(bundle.with_suffix(bundle.suffix + ".sha256").is_file())
        self.assertTrue((output / "CONTENTS.sha256").is_file())
        self.assertEqual(
            {(call["model"], call["mode"], call["weights_only"]) for call in runner.calls},
            {
                ("pretrained", "full", "true"),
                ("full", "zero", "true"),
                ("full", "full", "true"),
            },
        )
        differences = pd.read_csv(
            output
            / "post_holdout_explanatory_reports"
            / "post_holdout_explanatory_differences.csv"
        )
        self.assertTrue((differences["evidence_class"] == EVIDENCE_CLASS).all())
        self.assertTrue((differences["confirmatory_test"] == "not_performed").all())
        self.assertTrue((differences["p_value"] == "not_computed").all())

        source = output / "post_holdout_explanatory_frozen_v4_inputs"
        original = self.base / "successful_v4" / "weights" / "weights_rl_full_seed_1.csv"
        copied = source / "weights" / "weights_rl_full_seed_1.csv"
        self.assertEqual(sha256_file(original), sha256_file(copied))

    def test_archive_hash_mismatch_fails_before_policy_generation(self) -> None:
        definition = json.loads(self.contract.read_text(encoding="utf-8"))
        definition["successful_v4"]["archive_sha256"] = "0" * 64
        self.contract.write_text(json.dumps(definition), encoding="utf-8")
        runner = SyntheticRunner(self.realized)
        with self.assertRaisesRegex(SecondaryAblationError, "Wrong successful archive"):
            self._execute(runner)
        self.assertEqual(runner.calls, [])
        self.assertFalse((self.base / "post_holdout_explanatory_secondary_release").exists())

    def test_generated_period_mismatch_is_preserved_as_failed_release(self) -> None:
        runner = SyntheticRunner(self.realized, malformed=True)
        with self.assertRaisesRegex(SecondaryAblationError, "exact consumed-holdout period keys"):
            self._execute(runner)
        output = self.base / "post_holdout_explanatory_secondary_release"
        bundle = self.base / "post_holdout_explanatory_secondary_release.tar.gz"
        failure = json.loads(
            (output / "post_holdout_explanatory_release_manifest.json").read_text()
        )
        self.assertEqual(failure["release_status"], "failed_post_holdout_explanatory_batch")
        self.assertTrue(failure["successful_v4_archive_accessed"])
        self.assertTrue(bundle.is_file())
        self.assertTrue((output / "post_holdout_explanatory_command_logs").is_dir())

    def test_tar_failure_preserves_explanatory_logs_and_failed_label(self) -> None:
        runner = SyntheticRunner(self.realized)
        with patch(
            "publication_pipeline_draft.secondary_ablation_batch.deterministic_tar",
            side_effect=OSError("simulated compressor failure"),
        ):
            with self.assertRaisesRegex(SecondaryAblationError, "Failure logs were preserved"):
                self._execute(runner)
        output = self.base / "post_holdout_explanatory_secondary_release"
        bundle = self.base / "post_holdout_explanatory_secondary_release.tar.gz"
        failure = json.loads(
            (output / "post_holdout_explanatory_release_manifest.json").read_text()
        )
        self.assertEqual(failure["release_status"], "failed_post_holdout_explanatory_batch")
        self.assertIn("simulated compressor failure", failure["error"])
        self.assertTrue(any((output / "post_holdout_explanatory_command_logs").glob("*.stdout.txt")))
        self.assertFalse(bundle.exists())
        self.assertFalse(bundle.with_suffix(bundle.suffix + ".sha256").exists())

    def test_no_vine_mode_mismatch_fails_before_policy_generation(self) -> None:
        mode = self.no_vine_release / "seeds" / "seed_11" / "vine_observation_mode.txt"
        mode.write_text("full\n", encoding="utf-8")
        write_contents(self.no_vine_release)
        runner = SyntheticRunner(self.realized)
        with self.assertRaisesRegex(SecondaryAblationError, "expected 'zero'"):
            self._execute(runner)
        self.assertEqual(runner.calls, [])

    def test_legacy_full_release_without_mode_markers_is_allowed_and_replayed(self) -> None:
        for seed in self.full_seeds:
            directory = self.full_release / "seeds" / f"seed_{seed}"
            (directory / "vine_observation_mode.txt").unlink()
            report_path = directory / "sanity_no_holdout" / "sanity_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.pop("vine_observation_mode")
            report_path.write_text(json.dumps(report), encoding="utf-8")
        write_contents(self.full_release)

        runner = SyntheticRunner(self.realized)
        output, _, manifest = self._execute(runner)
        self.assertTrue(manifest["full_inference_replay_verified"])
        matched = pd.read_csv(
            output
            / "post_holdout_explanatory_reports"
            / "post_holdout_explanatory_matched_design_verification.csv"
        )
        row = matched.loc[
            matched["check"] == "protocol:full_vine_observation_mode_evidence"
        ].iloc[0]
        self.assertEqual(row["status"], "pass")
        self.assertIn("legacy_missing_allowed_by_frozen_contract", row["full_value"])

    def test_legacy_full_permission_never_accepts_a_conflicting_mode(self) -> None:
        report_path = (
            self.full_release
            / "seeds"
            / f"seed_{self.full_seeds[0]}"
            / "sanity_no_holdout"
            / "sanity_report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["vine_observation_mode"] = "zero"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        write_contents(self.full_release)

        runner = SyntheticRunner(self.realized)
        with self.assertRaisesRegex(SecondaryAblationError, "sanity mode differs"):
            self._execute(runner)
        self.assertEqual(runner.calls, [])

    def test_live_full_inference_drift_blocks_all_ablation_scoring(self) -> None:
        runner = SyntheticRunner(self.realized, replay_drift=True)
        with self.assertRaisesRegex(
            SecondaryAblationError, "does not reproduce archived v4 full weights"
        ):
            self._execute(runner)
        self.assertTrue(runner.calls)
        self.assertTrue(
            all(call["label"].startswith("full_v4_inference_replay") for call in runner.calls)
        )

    def test_actor_capacity_mismatch_fails_before_policy_generation(self) -> None:
        path = (
            self.no_vine_release
            / "seeds"
            / "seed_11"
            / "sanity_no_holdout"
            / "checkpoint_integrity.csv"
        )
        frame = pd.read_csv(path)
        frame.loc[frame["model"] == "full", "actor_parameters"] = 9999
        frame.to_csv(path, index=False)
        write_contents(self.no_vine_release)
        runner = SyntheticRunner(self.realized)
        with self.assertRaisesRegex(
            SecondaryAblationError, "Matched-capacity/training-budget validation failed"
        ):
            self._execute(runner)
        self.assertEqual(runner.calls, [])

    def test_nonfinite_archived_replay_value_is_fatal(self) -> None:
        contract = Contract.read(self.evaluation_contract)
        realized = self.realized.copy()
        for column in ["decision_date", "holding_end_date"]:
            realized[column] = pd.to_datetime(realized[column])
        weights = self._weights(0.5)
        for column in ["decision_date", "holding_end_date"]:
            weights[column] = pd.to_datetime(weights[column])
        current = score_strategy("probe", weights, realized, ["A", "B"], contract)
        archived = current.copy()
        archived.loc[0, "gross_return"] = float("nan")
        with self.assertRaisesRegex(SecondaryAblationError, "non-finite gross_return"):
            verify_archived_scores({"probe": current}, archived, ["A", "B"], 1e-12)


if __name__ == "__main__":
    unittest.main()
