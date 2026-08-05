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

from diagnostic_artifacts import synthetic_artifacts, training_artifacts  # noqa: E402


class DiagnosticArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_synthetic_adapter_exports_summary_and_figures(self) -> None:
        source = self.base / "synthetic"
        source.mkdir()
        fidelity = pd.DataFrame(
            {
                "asset": ["A", "B"],
                "historical_mean": [0.01, 0.02], "synthetic_mean": [0.011, 0.019],
                "historical_sd": [0.04, 0.05], "synthetic_sd": [0.041, 0.049],
                "historical_q05": [-0.06, -0.07], "synthetic_q05": [-0.061, -0.069],
                "historical_cvar05": [-0.08, -0.09], "synthetic_cvar05": [-0.081, -0.089],
                "pass_marginals": [True, True],
            }
        )
        correlation = pd.DataFrame(
            {"asset_i": ["A"], "asset_j": ["B"], "historical_correlation": [0.4],
             "synthetic_correlation": [0.39], "pass_correlation": [True]}
        )
        tail = pd.DataFrame(
            {"asset_i": ["A"], "asset_j": ["B"], "historical_lower_tail": [0.2],
             "synthetic_lower_tail": [0.21], "historical_tail_events": [6],
             "historical_joint_tail_events": [1], "pass_lower_tail": [True]}
        )
        temporal = pd.DataFrame(
            {"asset": ["A", "B"], "historical_lag1": [0.1, -0.1],
             "synthetic_lag1": [0.09, -0.11], "historical_squared_lag1": [0.2, 0.3],
             "synthetic_squared_lag1": [0.21, 0.29], "pass_temporal": [True, True]}
        )
        fidelity.to_csv(source / "fidelity_metrics.csv", index=False)
        correlation.to_csv(source / "correlation_comparison.csv", index=False)
        tail.to_csv(source / "tail_dependence_comparison.csv", index=False)
        temporal.to_csv(source / "temporal_dependence.csv", index=False)
        pd.DataFrame({"regime": ["historical"], "asset": ["A"], "mean": [0.01]}).to_csv(source / "summary_statistics.csv", index=False)
        pd.DataFrame({"asset": ["A"], "var05": [-0.06]}).to_csv(source / "tail_risk.csv", index=False)
        output = self.base / "synthetic_output"
        result = synthetic_artifacts(source, output)
        self.assertEqual(result["gate_summary"][0]["passed"], 2)
        self.assertTrue((output / "tables" / "table_s01_synthetic_gate_summary.csv").is_file())
        self.assertTrue((output / "figures" / "figure_s02_dependence_fidelity.pdf").is_file())

    def test_training_adapter_requires_and_aggregates_all_seeds(self) -> None:
        rl_runs = self.base / "rl_runs"
        rl_runs.mkdir()
        status_rows = []
        for seed in [101, 102]:
            run = rl_runs / f"seed_{seed}"
            (run / "sanity_no_holdout").mkdir(parents=True)
            episodes = np.arange(1, 61)
            pd.DataFrame(
                {
                    "stage": "pretrain", "episode": episodes,
                    "reward": np.sin(episodes / 10) + seed / 10000,
                    "terminal_wealth": 100000 + 100 * episodes,
                    "mean_turnover": 0.4 + 0.001 * episodes,
                    "mean_cvar": 0.06, "mean_gross_exposure": 1.2,
                }
            ).to_csv(run / "training_episode_metrics.csv", index=False)
            updates = np.arange(100, 1100, 100)
            pd.DataFrame(
                {
                    "stage": "pretrain", "update": updates, "critic_loss": 1 / updates,
                    "actor_loss": 0.2 + updates * 0, "twin_q_gap": 0.01,
                    "actor_grad_norm": 0.02, "critic_grad_norm": 0.03,
                }
            ).to_csv(run / "training_update_metrics.csv", index=False)
            pd.DataFrame(
                {"metric": ["mean_turnover"], "value": [0.45], "pass": [True]}
            ).to_csv(run / "pretraining_behavior_gate.csv", index=False)
            pd.DataFrame({"model": ["full"], "mean_terminal_wealth": [110000]}).to_csv(
                run / "sanity_no_holdout" / "policy_summary.csv", index=False
            )
            pd.DataFrame(
                {"model": ["full"], "perturbation": ["zero_vine"], "median_action_l1_change": [0.1]}
            ).to_csv(run / "sanity_no_holdout" / "state_sensitivity_summary.csv", index=False)
            pd.DataFrame(
                {"stage": ["finetune_validation"], "pass": [1], "reward": [0.1],
                 "terminal_wealth": [110000], "mean_turnover": [0.4]}
            ).to_csv(run / "finetune_validation_metrics.csv", index=False)
            pd.DataFrame(
                {"stage": ["selection_fit", "final_refit"], "pass": [1, 1],
                 "position": [1, 1], "original_episode": [1, 1]}
            ).to_csv(run / "finetune_episode_schedule.csv", index=False)
            (run / "finetune_selection.txt").write_text("selected_pass=1\n", encoding="utf-8")
            (run / "run_manifest.rds").write_bytes(b"fixture manifest")
            checkpoint_records = []
            for model in ["pretrained", "full"]:
                checkpoint = run / f"td3_lstm_vine_{model}.pt"
                checkpoint.write_bytes(f"checkpoint-{seed}-{model}".encode())
                checkpoint_records.append(
                    {"model": model, "path": str(checkpoint),
                     "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     "architecture_match": True, "all_checkpoint_tensors_finite": True,
                     "tensor_count": 10, "tensor_elements": 100}
                )
            pd.DataFrame(checkpoint_records).to_csv(
                run / "sanity_no_holdout" / "checkpoint_integrity.csv", index=False
            )
            (run / "sanity_no_holdout" / "sanity_report.json").write_text(
                json.dumps(
                    {"protocol": "training prefix only", "seed": seed, "episodes": 2,
                     "episode_length": 24, "obs_dim": 88, "action_dim": 7,
                     "vine_dim": 63, "warnings": [], "diagnostic_notes": [],
                     "publication_behavior_pass": True, "overall_pass": True}
                ), encoding="utf-8"
            )
            pd.DataFrame(
                {"path": ["/repo/data/input.csv"], "md5": ["a" * 32]}
            ).to_csv(run / "data_hashes.csv", index=False)
            pd.DataFrame(
                {"path": ["/repo/rl/train_rl.r"], "md5": ["b" * 32]}
            ).to_csv(run / "code_hashes.csv", index=False)
            status_rows.append(
                {"seed": seed, "output_dir": str(run), "training_status": 0,
                 "sanity_status": 0, "no_holdout_gate_pass": True}
            )
        pd.DataFrame(status_rows).to_csv(rl_runs / "seed_sweep_status.csv", index=False)
        output = self.base / "training_output"
        result = training_artifacts(rl_runs, output, expected_seeds=2)
        self.assertEqual(result["gate_pass_count"], 2)
        self.assertEqual(result["checkpoint_count"], 4)
        self.assertTrue((output / "figures" / "figure_t01_pretraining_stability.pdf").is_file())
        self.assertTrue((output / "raw" / "training_update_metrics_all_seeds.csv").is_file())
        self.assertTrue((output / "tables" / "table_t09_code_data_hash_consensus.csv").is_file())


if __name__ == "__main__":
    unittest.main()
