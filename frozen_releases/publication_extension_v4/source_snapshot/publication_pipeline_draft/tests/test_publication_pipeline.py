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

from publication_pipeline import (  # noqa: E402
    Contract,
    ProtocolError,
    empirical_metrics,
    read_realized_panel,
    run_pipeline,
    score_strategy,
    validate_weight_matrix,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublicationPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        dates = pd.date_range("2024-01-31", periods=11, freq="ME")
        self.realized = pd.DataFrame(
            {
                "window_id": "w1",
                "decision_date": dates[:-1],
                "holding_end_date": dates[1:],
                "trading_days": [21, 20, 22, 21, 20, 22, 21, 20, 22, 5],
                "is_complete_period": [True] * 9 + [False],
                "g_A": [1.02, 0.98, 1.03, 1.01, 0.99, 1.02, 1.01, 0.97, 1.04, 1.01],
                "g_B": [1.01, 1.02, 0.99, 1.00, 1.03, 0.98, 1.02, 1.01, 0.99, 1.00],
            }
        )
        self.realized_path = self.base / "realized.csv"
        self.realized.to_csv(self.realized_path, index=False)
        self.contract = {
            "schema_version": 1,
            "evaluation_id": "test",
            "expected_locked_periods_per_window": 10,
            "minimum_complete_periods_per_window": 9,
            "primary_sample_scope": "complete_periods",
            "periods_per_year": 12,
            "initial_wealth": 100000.0,
            "net_exposure": 1.0,
            "gross_leverage": 1.5,
            "max_long_weight": 0.9,
            "max_short_weight": 0.5,
            "turnover_cost": 0.001,
            "annual_short_borrow_rate": 0.03,
            "annual_cash_borrow_rate": 0.02,
            "annual_risk_free_rate": 0.0,
            "crra_gamma": 2.0,
            "primary_benchmark_id": "equal_weight",
            "primary_strategy_id": "ensemble",
            "primary_superiority_test": "one_sided_paired_moving_block_bootstrap_crra",
            "primary_superiority_alpha": 0.05,
            "secondary_multiplicity_control": "holm_within_primary_vs_alternative_family",
            "bootstrap_replications": 999,
            "bootstrap_block_length": 2,
            "inference_seed": 123,
            "weight_tolerance": 1e-6,
            "require_weight_log_hashes": True,
            "require_checkpoint_hash_for_trained_models": False,
            "require_code_and_config_hashes": False,
            "predeclared_ensembles": [
                {
                    "strategy_id": "ensemble",
                    "label": "Policy ensemble",
                    "method": "Policy",
                    "ensemble_group": "passing_policy",
                    "minimum_members": 2,
                    "include_main": True,
                    "include_inference": True,
                }
            ],
        }
        self.contract_path = self.base / "contract.json"
        self.contract_path.write_text(json.dumps(self.contract), encoding="utf-8")

        self.weight_paths: list[Path] = []
        for seed, first_weights in [(1, (0.70, 0.30)), (2, (0.50, 0.50))]:
            weights = self.realized[["window_id", "decision_date", "holding_end_date"]].copy()
            weights["w_A"] = [first_weights[0], 0.60, 0.55, 0.65, 0.58, 0.62, 0.57, 0.61, 0.59, 0.63]
            weights["w_B"] = 1.0 - weights["w_A"]
            path = self.base / f"seed_{seed}.csv"
            weights.to_csv(path, index=False)
            self.weight_paths.append(path)

        fields = [
            "strategy_id", "label", "method", "seed", "role", "completed",
            "gate_pass", "ensemble_group", "include_main", "include_inference",
            "report_seed_distribution", "weight_log_path", "weight_log_sha256",
            "checkpoint_path", "checkpoint_sha256", "config_sha256", "code_sha256",
            "train_seconds", "evaluation_seconds", "notes",
        ]
        rows = [
            {
                "strategy_id": "equal_weight", "label": "Equal weight", "method": "Equal weight",
                "seed": "", "role": "benchmark", "completed": True, "gate_pass": True,
                "ensemble_group": "", "include_main": True, "include_inference": True,
                "report_seed_distribution": False, "weight_log_path": "GENERATE_EQUAL_WEIGHT",
                "weight_log_sha256": "", "checkpoint_path": "not_applicable",
                "checkpoint_sha256": "not_applicable", "config_sha256": "", "code_sha256": "",
                "train_seconds": 0, "evaluation_seconds": 0, "notes": "",
            }
        ]
        for seed, path in enumerate(self.weight_paths, 1):
            rows.append(
                {
                    "strategy_id": f"seed_{seed}", "label": f"Policy seed {seed}", "method": "Policy",
                    "seed": seed, "role": "proposed", "completed": True, "gate_pass": True,
                    "ensemble_group": "passing_policy", "include_main": False,
                    "include_inference": False, "report_seed_distribution": True,
                    "weight_log_path": path.name, "weight_log_sha256": digest(path),
                    "checkpoint_path": "ignored", "checkpoint_sha256": "",
                    "config_sha256": "", "code_sha256": "", "train_seconds": 10,
                    "evaluation_seconds": 1, "notes": "",
                }
            )
        self.manifest_path = self.base / "strategies.csv"
        pd.DataFrame(rows, columns=fields).to_csv(self.manifest_path, index=False)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_complete_period_scope_and_asset_detection(self) -> None:
        contract = Contract.read(self.contract_path)
        frame, assets = read_realized_panel(self.realized_path, contract)
        self.assertEqual(assets, ["A", "B"])
        self.assertEqual(int(frame["is_complete_period"].sum()), 9)

    def test_common_score_matches_manual_first_period(self) -> None:
        contract = Contract.read(self.contract_path)
        realized, assets = read_realized_panel(self.realized_path, contract)
        weights = realized[["window_id", "decision_date", "holding_end_date"]].copy()
        weights["w_A"] = 0.7
        weights["w_B"] = 0.3
        scored = score_strategy("manual", weights, realized, assets, contract)
        turnover = abs(0.7 - 0.5) + abs(0.3 - 0.5)
        gross = 1 + 0.7 * 0.02 + 0.3 * 0.01
        expected_net = gross * np.exp(-0.001 * turnover)
        self.assertAlmostEqual(scored.loc[0, "net_return"], expected_net - 1, places=12)

    def test_position_cap_is_enforced(self) -> None:
        contract = Contract.read(self.contract_path)
        with self.assertRaises(ProtocolError):
            validate_weight_matrix(np.array([[0.95, 0.05]]), "bad", contract)

    def test_drawdown_and_certainty_equivalent_metrics(self) -> None:
        contract = Contract.read(self.contract_path)
        group = pd.DataFrame(
            {
                "decision_date": pd.date_range("2024-01-31", periods=3, freq="ME"),
                "net_return": [0.10, -0.10, 0.05],
                "gross_return": [0.10, -0.10, 0.05],
                "turnover": [0.0, 0.0, 0.0],
                "gross_exposure": [1.0, 1.0, 1.0],
                "short_notional": [0.0, 0.0, 0.0],
                "transaction_cost": [0.0, 0.0, 0.0],
                "financing_cost": [0.0, 0.0, 0.0],
            }
        )
        metrics = empirical_metrics(group, contract)
        self.assertAlmostEqual(metrics["max_drawdown"], 0.10, places=12)
        self.assertTrue(np.isfinite(metrics["annualized_certainty_equivalent_return"]))
        self.assertAlmostEqual(metrics["implementation_drag_total_return"], 0.0, places=12)

    def test_end_to_end_outputs_ensemble_and_primary_scope(self) -> None:
        output = self.base / "results"
        run_pipeline(self.contract_path, self.realized_path, self.manifest_path, output)
        self.assertTrue((output / "run_manifest.json").is_file())
        main = pd.read_csv(output / "tables" / "table_01_oos_performance.csv")
        self.assertEqual(set(main["strategy_id"]), {"equal_weight", "ensemble"})
        self.assertTrue((main["sample_scope"] == "complete_periods").all())
        self.assertTrue((main["observations"] == 9).all())
        scored = pd.read_csv(output / "raw" / "scored_monthly_panel.csv")
        ensemble = scored[scored["strategy_id"] == "ensemble"]
        self.assertAlmostEqual(float(ensemble.iloc[0]["w_A"]), 0.60)
        self.assertTrue((output / "figures" / "figure_03_risk_return.pdf").is_file())
        self.assertTrue((output / "figures" / "figure_07_primary_utility_effects.pdf").is_file())
        self.assertTrue((output / "tables" / "table_06_monthly_net_returns.csv").is_file())
        self.assertTrue((output / "tables" / "table_09_return_distribution.csv").is_file())
        self.assertTrue((output / "tables" / "table_10_primary_effects.csv").is_file())
        constraints = pd.read_csv(output / "tables" / "table_11_constraint_audit.csv")
        self.assertTrue(constraints["constraint_pass"].all())
        self.assertTrue(
            (output / "tables" / "table_12_cost_and_borrow_sensitivity.csv").is_file()
        )
        self.assertTrue(
            (output / "tables" / "table_13_ensemble_size_sensitivity.csv").is_file()
        )
        decision = json.loads(
            (output / "tables" / "primary_superiority_decision.json").read_text()
        )
        self.assertEqual(decision["primary_benchmark_id"], "equal_weight")

    def test_locked_output_cannot_be_overwritten(self) -> None:
        output = self.base / "results"
        run_pipeline(self.contract_path, self.realized_path, self.manifest_path, output)
        with self.assertRaises(ProtocolError):
            run_pipeline(self.contract_path, self.realized_path, self.manifest_path, output)


if __name__ == "__main__":
    unittest.main()
