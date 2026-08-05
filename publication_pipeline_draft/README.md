# Frozen publication-evaluation draft

This directory is intentionally outside `rl/`, `eval/`, `config/`, and the
root-level R launchers. The running seed sweep neither sources nor hashes it.
Do not move these files into the live pipeline until the seed sweep has
finished and its artifacts have been copied to immutable storage.

The draft separates policy generation from performance measurement:

1. Every model or benchmark emits **weights only** for each locked decision.
2. `build_realized_panel.R` exports the one common realized asset-return panel.
3. `publication_pipeline.py` validates dates and mandate constraints, applies
   identical costs, constructs the predeclared multi-seed RL ensemble, computes
   metrics and paired inference, and writes paper-ready CSV/LaTeX/PNG/PDF files.
4. No missing metric, run, hash, or strategy is replaced by a simulated or
   hard-coded fallback.

## Primary estimand fixed before holdout inspection

- Proposed strategy: the date-by-date mean weights of all full-model TD3 seeds
  that passed the no-holdout gate. Averaging is an investable ensemble and
  preserves the convex net/gross/position constraints.
- Individual seeds: reported as algorithmic variability, not independent
  market samples and not pooled into market-significance tests.
- Primary calendar: complete monthly holding periods only. The locked panel
  still retains and reports any shortened terminal period under the
  `locked_all` robustness scope.
- Benchmark for the White-style reality check: equal weight, declared in the
  contract rather than inferred from file order.

## Expected inputs

Copy the examples in `config/` after the sweep:

- `evaluation_contract.example.json`: immutable economic/statistical contract.
- `strategy_manifest.example.csv`: one row per completed weight-producing run.
- `realized_asset_gross.csv`: one row per locked holding period with columns
  `window_id`, `decision_date`, `holding_end_date`, `trading_days`,
  `is_complete_period`, and `g_<ASSET>`.
- Weight logs: the same key columns plus `w_<ASSET>`.

Paths in the strategy manifest are resolved relative to that manifest. A
`weight_log_sha256` should be filled after each log is frozen. Checkpoint hashes
are required for trained strategies; deterministic rules such as equal weight
use `not_applicable`.

## Run after the sweep

```powershell
python publication_pipeline_draft/publication_pipeline.py `
  --contract publication_eval/evaluation_contract.json `
  --realized publication_eval/realized_asset_gross.csv `
  --strategies publication_eval/strategy_manifest.csv `
  --output publication_eval/results
```

The output directory must not already exist. This prevents an evaluation rerun
from silently overwriting the first locked result.

## Generated artifacts

Core raw exports:

- `raw/scored_monthly_panel.csv`
- `raw/validated_strategy_manifest.csv`
- `raw/input_hashes.csv`
- `raw/protocol_checks.csv`

Paper tables:

- `tables/table_01_oos_performance.csv` and `.tex`
- `tables/table_02_seed_robustness.csv` and `.tex`
- `tables/table_03_paired_inference.csv` and `.tex`
- `tables/table_04_economic_implementation.csv` and `.tex`
- `tables/table_05_computation.csv` and `.tex`

Core figures (both PNG and PDF):

- wealth and drawdown paths;
- risk-return map;
- primary-RL allocation heatmap;
- exposure, short-notional, and turnover path;
- seed-robustness distributions when at least two individual seeds exist.

`config/paper_artifact_catalog.csv` lists the broader main-text and appendix
artifact plan, including synthetic-data, training, ablation, and sensitivity
outputs that require their own completed experiments.

Synthetic and training artifacts can be built without opening the holdout:

```powershell
python publication_pipeline_draft/diagnostic_artifacts.py `
  --synthetic-diagnostics data/synthetic_diagnostics `
  --rl-runs data/rl_runs `
  --expected-seeds 20 `
  --output publication_training_artifacts
```

Follow `POST_SWEEP_RUNBOOK.md` for the freeze, benchmark, locked-batch, and
interpretation order.

Freeze the accepted training release on the machine that contains the exact
20 seed directories and the exact training data/source snapshot:

```bash
python publication_pipeline_draft/freeze_training_release.py \
  --repo-root . \
  --rl-runs data/rl_runs \
  --diagnostics-archive training_artifacts_20seeds.tar.gz \
  --expected-seeds 20 \
  --output frozen_releases/training_schema5_v1 \
  --bundle frozen_releases/training_schema5_v1.tar.gz
```

The freezer is holdout-blind and fail-closed. It re-hashes all recorded
training source and data files, checks every seed artifact and checkpoint
against the aggregate diagnostics, copies the exact source plus seed evidence,
and refuses an existing output. Large training data are verified by default
but only copied when `--copy-training-data` is supplied. Store the resulting
bundle and its `.sha256` sidecar in immutable storage before any OOS run.

The training aggregation is fail-closed. For every seed it now requires and
hashes the episode/update logs, pre-training gate, fine-tuning validation and
schedule, selection record, run manifest, both checkpoints, checkpoint
integrity table, sanity report, and code/data hash tables. It rejects a batch
when checkpoint hashes fail or code/data hashes differ across seeds.

## Verification

The local machine currently has no R runtime, so the R adapter can only be
reviewed statically here. The Python evaluator has executable unit tests:

```powershell
python -m unittest discover -s publication_pipeline_draft/tests -v
```
