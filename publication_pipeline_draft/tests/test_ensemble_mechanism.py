from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_ensemble_mechanism import (  # noqa: E402
    AnalysisContract,
    ExplanatoryAnalysisError,
    k_seed_sensitivity,
    load_completed_batch,
    run_analysis,
    score_weight_path,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EnsembleMechanismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.batch = self.root / "locked_batch"
        (self.batch / "weights").mkdir(parents=True)
        (self.batch / "inputs").mkdir()
        (self.batch / "benchmark_weights").mkdir()
        (self.batch / "publication_results" / "raw").mkdir(parents=True)

        dates = pd.date_range("2024-01-31", periods=5, freq="ME")
        self.realized = pd.DataFrame({
            "window_id": "w1",
            "decision_date": dates[:-1],
            "holding_end_date": dates[1:],
            "trading_days": [21, 20, 22, 21],
            "is_complete_period": [True, True, True, True],
            "g_A": [1.10, 0.95, 1.04, 1.02],
            "g_B": [0.98, 1.08, 0.99, 1.03],
        })
        self.realized.to_csv(self.batch / "inputs" / "realized_asset_gross.csv", index=False)
        self.contract_raw = {
            "schema_version": 1,
            "analysis_classification": "post_holdout_explanatory",
            "confirmatory_use_permitted": False,
            "expected_seed_count": 2,
            "seed_strategy_prefix": "seed_",
            "ensemble_strategy_id": "ensemble",
            "sample_scope": "complete_periods",
            "periods_per_year": 12,
            "initial_wealth": 100000.0,
            "arithmetic_tolerance": 1e-10,
            "k_seed_sizes": [1, 2],
            "bootstrap_replications": 100,
            "analysis_seed": 99,
        }
        self.contract_path = self.root / "analysis_contract.json"
        self.contract_path.write_text(json.dumps(self.contract_raw), encoding="utf-8")
        self.contract = AnalysisContract.read(self.contract_path)
        self.economics = {
            "schema_version": 1,
            "evaluation_id": "w1",
            "net_exposure": 1.0,
            "gross_leverage": 1.5,
            "max_long_weight": 0.9,
            "max_short_weight": 0.5,
            "weight_tolerance": 1e-6,
            "turnover_cost": 0.001,
            "annual_short_borrow_rate": 0.03,
            "annual_cash_borrow_rate": 0.02,
            "crra_gamma": 2.0,
        }
        (self.batch / "benchmark_weights" / "benchmark_contract.json").write_text(
            json.dumps(self.economics), encoding="utf-8")
        (self.batch / "locked_batch_manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "status": "complete",
            "holdout_accessed": True,
            "full_policy_count": 2,
        }), encoding="utf-8")
        (self.batch / "publication_results" / "run_manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "primary_strategy_id": "ensemble",
        }), encoding="utf-8")

        self.seed_weights = [
            np.array([[0.75, 0.25], [0.65, 0.35], [0.70, 0.30], [0.60, 0.40]]),
            np.array([[0.30, 0.70], [0.40, 0.60], [0.35, 0.65], [0.45, 0.55]]),
        ]
        scored_parts = []
        manifest_rows = []
        for seed, weights in enumerate(self.seed_weights, 1):
            strategy_id = f"seed_{seed}"
            path = self.batch / "weights" / f"weights_{strategy_id}.csv"
            weight_frame = self.realized[[
                "window_id", "decision_date", "holding_end_date"]].copy()
            weight_frame[["w_A", "w_B"]] = weights
            weight_frame.to_csv(path, index=False)
            scored = score_weight_path(
                weights, self.realized, ["A", "B"], self.economics, self.contract)
            scored.insert(0, "strategy_id", strategy_id)
            scored_parts.append(scored)
            manifest_rows.append({
                "strategy_id": strategy_id,
                "label": strategy_id,
                "method": "Policy",
                "seed": seed,
                "role": "proposed",
                "completed": True,
                "gate_pass": True,
                "ensemble_group": "passing",
                "include_main": False,
                "include_inference": False,
                "report_seed_distribution": True,
                "weight_log_path": f"weights/{path.name}",
                "weight_log_sha256": digest(path),
                "checkpoint_path": f"checkpoints/{strategy_id}.pt",
                "checkpoint_sha256": f"{seed:064x}",
                "config_sha256": "a" * 64,
                "code_sha256": "b" * 64,
                "train_seconds": 1,
                "evaluation_seconds": 1,
                "notes": "",
            })
        ensemble_weights = np.mean(self.seed_weights, axis=0)
        ensemble = score_weight_path(
            ensemble_weights, self.realized, ["A", "B"], self.economics, self.contract)
        ensemble.insert(0, "strategy_id", "ensemble")
        scored_parts.append(ensemble)
        equal = score_weight_path(
            np.full((4, 2), 0.5), self.realized, ["A", "B"],
            self.economics, self.contract)
        equal.insert(0, "strategy_id", "equal_weight")
        scored_parts.append(equal)
        pd.concat(scored_parts, ignore_index=True).to_csv(
            self.batch / "publication_results" / "raw" / "scored_monthly_panel.csv",
            index=False)
        manifest_rows.insert(0, {
            "strategy_id": "equal_weight",
            "label": "Equal weight",
            "method": "Equal weight",
            "seed": "",
            "role": "benchmark",
            "completed": True,
            "gate_pass": True,
            "ensemble_group": "",
            "include_main": True,
            "include_inference": True,
            "report_seed_distribution": False,
            "weight_log_path": "GENERATE_EQUAL_WEIGHT",
            "weight_log_sha256": "",
            "checkpoint_path": "not_applicable",
            "checkpoint_sha256": "not_applicable",
            "config_sha256": "",
            "code_sha256": "",
            "train_seconds": 0,
            "evaluation_seconds": 0,
            "notes": "",
        })
        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(self.batch / "strategy_manifest.csv", index=False)
        manifest.to_csv(
            self.batch / "publication_results" / "raw" / "validated_strategy_manifest.csv",
            index=False)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_contract_forbids_confirmatory_reclassification(self) -> None:
        invalid = dict(self.contract_raw)
        invalid["confirmatory_use_permitted"] = True
        path = self.root / "invalid_contract.json"
        path.write_text(json.dumps(invalid), encoding="utf-8")
        with self.assertRaises(ExplanatoryAnalysisError):
            AnalysisContract.read(path)

    def test_completed_batch_verifies_exact_arithmetic_ensemble(self) -> None:
        data = load_completed_batch(self.batch, self.contract)
        self.assertEqual(data.seed_ids, ["seed_1", "seed_2"])
        np.testing.assert_allclose(
            data.ensemble_weights, np.mean(data.seed_weights, axis=0), atol=1e-12)
        self.assertLess(
            float(data.verification["maximum_ensemble_weight_error"].max()), 1e-12)

    def test_arithmetic_ensemble_mismatch_is_fatal(self) -> None:
        path = self.batch / "publication_results" / "raw" / "scored_monthly_panel.csv"
        scored = pd.read_csv(path)
        target = scored.index[scored["strategy_id"] == "ensemble"][0]
        scored.loc[target, "w_A"] += 0.01
        scored.to_csv(path, index=False)
        with self.assertRaises(ExplanatoryAnalysisError):
            load_completed_batch(self.batch, self.contract)

    def test_drift_aware_turnover_uses_realized_post_return_weights(self) -> None:
        weights = self.seed_weights[0]
        target = score_weight_path(
            weights, self.realized, ["A", "B"], self.economics, self.contract,
            turnover_mode="target_to_target")
        drift = score_weight_path(
            weights, self.realized, ["A", "B"], self.economics, self.contract,
            turnover_mode="drift_aware")
        gross_first = 1.0 + np.dot(weights[0], np.array([0.10, -0.02]))
        post = weights[0] * np.array([1.10, 0.98]) / gross_first
        expected = np.abs(weights[1] - post).sum()
        self.assertAlmostEqual(float(drift.loc[1, "turnover"]), expected, places=12)
        self.assertNotAlmostEqual(
            float(drift.loc[1, "turnover"]), float(target.loc[1, "turnover"]), places=8)

    def test_k_seed_bootstrap_is_deterministic(self) -> None:
        data = load_completed_batch(self.batch, self.contract)
        first = k_seed_sensitivity(data, self.contract)
        second = k_seed_sensitivity(data, self.contract)
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_frame_equal(first[2], second[2])

    def test_end_to_end_outputs_are_explicitly_explanatory(self) -> None:
        output = self.root / "explanatory_output"
        run_analysis(self.batch, self.contract_path, output)
        self.assertTrue((output / "run_manifest.json").is_file())
        self.assertTrue((output / "figures" / "figure_exploratory_k_seed_sensitivity.pdf").is_file())
        table = pd.read_csv(
            output / "tables" / "explanatory_ensemble_mechanism_summary.csv")
        self.assertTrue((table["analysis_classification"] == "post_holdout_explanatory").all())
        self.assertFalse(table["confirmatory_use_permitted"].astype(bool).any())
        run_manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(run_manifest["status"], "complete")
        self.assertFalse(run_manifest["confirmatory_use_permitted"])
        with self.assertRaises(ExplanatoryAnalysisError):
            run_analysis(self.batch, self.contract_path, output)


if __name__ == "__main__":
    unittest.main()
