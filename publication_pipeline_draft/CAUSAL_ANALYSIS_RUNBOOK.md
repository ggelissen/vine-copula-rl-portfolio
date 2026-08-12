# Causal analysis v1: frozen post-holdout explanatory workflow

> Execution is paused after the disclosed 70/130 v2 diagnostic sweep.  Complete
> and audit `PUBLICATION_EXTENSION_V3_RECOVERY.md` first, then create a new
> versioned causal-analysis-plan release bound to the v3 extension.  Do not use
> the v2 paths below for new policy replay.

For the authorized mixed-revision recovery, substitute:

- jobs: `protocol_manifests/causal_jobs_v2_v3_merged.csv`;
- status: `protocol_manifests/causal_sweep_status_v2_v3_merged.csv`;
- audit: `analysis_outputs/causal_sweep_audit_v2_v3_merged`;
- current extension: `frozen_releases/publication_extension_v3_retry`;
- carried extension: `frozen_releases/publication_extension_v2`.

Freeze the outcome-blind analysis plan with both revision sources before policy
replay:

```bash
"$PYTHON" -m publication_pipeline_draft.freeze_causal_analysis_plan \
  --repo-root . \
  --extension-release frozen_releases/publication_extension_v3_retry \
  --carried-extension-release frozen_releases/publication_extension_v2 \
  --operational-merge-manifest protocol_manifests/causal_v2_v3_operational_merge.json \
  --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
  --output frozen_releases/causal_analysis_v2_v3_merged \
  --archive frozen_releases/causal_analysis_v2_v3_merged.tar.gz
```

This workflow evaluates the 13-by-10 causal training design. It deliberately
reuses the already consumed 24-month main holdout and therefore cannot create a
new confirmatory result. Its purpose is mechanism attribution: which model
components and algorithm choices explain the observed main-model behaviour.

The ten seeds measure training/optimization variability. They are not ten
independent market samples. All statistical market-path inference is paired by
month on the 13 weight-space ensemble paths.

## 0. Do not disturb the live training sweep

Do not pull, checkout, edit the frozen extension, or restart the live job while
the 130-job sweep is running. The new analysis files are not consumed by the
trainer. Synchronize them only after the sweep has completed and its status and
logs have been preserved.

Set the pinned runtimes after synchronization:

```bash
cd /gabirel/copula-portfolio-clean
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
export PYTHON=/gabirel/miniforge3/bin/python3
export TRAIN_PYTHON=/gabirel/miniforge3/envs/vine-rl/bin/python
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python
export RSCRIPT=/gabirel/miniforge3/bin/Rscript
```

## 1. Validate the prospective analysis contract

Run this before opening or summarizing any causal policy result:

```bash
"$PYTHON" -m compileall -q publication_pipeline_draft
"$PYTHON" -m pytest -q publication_pipeline_draft/tests

"$PYTHON" -m publication_pipeline_draft.causal_analysis_contract validate \
  --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json

"$PYTHON" -m publication_pipeline_draft.causal_analysis_contract materialize \
  --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
  --output protocol_manifests/causal_contrast_plan_v1.csv
```

Expected: 13 experiments, 8 primary component contrasts, 4 exploratory
algorithm contrasts, and 10 matched seeds.

## 2. Freeze the analysis plan before causal evaluation

This freezer reads method/code only. It does not read checkpoints, target
weights, realized returns, or results.

```bash
"$PYTHON" -m publication_pipeline_draft.freeze_causal_analysis_plan \
  --repo-root . \
  --extension-release frozen_releases/publication_extension_v2 \
  --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
  --output frozen_releases/causal_analysis_v1 \
  --archive frozen_releases/causal_analysis_v1.tar.gz

(
  cd frozen_releases/causal_analysis_v1
  sha256sum -c CONTENTS.sha256
)
(
  cd frozen_releases
  sha256sum -c causal_analysis_v1.tar.gz.sha256
)
```

Do not edit the frozen release. Any operational correction requires a new
version and an explicit statement that the scientific estimand did not change.

## 3. Audit all 130 completed training jobs

If the sweep was sharded, merge all shard statuses first using
`merge_causal_sweep_status.py`. Then run:

```bash
"$TRAIN_PYTHON" -m publication_pipeline_draft.audit_causal_sweep \
  --jobs protocol_manifests/causal_jobs_v2.csv \
  --status protocol_manifests/causal_sweep_status_v2.csv \
  --repo-root . \
  --output analysis_outputs/causal_sweep_audit_v2

(
  cd analysis_outputs/causal_sweep_audit_v2
  test "$(wc -l < checkpoint_audit.csv)" -eq 131
)
```

All 130 jobs must pass. Missing/failed seeds cannot be substituted or silently
dropped. Preserve a failed audit directory and use a versioned retry path.

## 4. Replay every audited checkpoint into target weights

This is the point at which the consumed holdout is accessed for the causal
study. It is explicitly post-holdout explanatory. Two CPU workers are a safe
default because each R evaluation also performs vine-state work; no training GPU
is needed.

```bash
"$PYTHON" -m publication_pipeline_draft.generate_causal_policy_weights \
  --repo-root . \
  --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
  --analysis-release frozen_releases/causal_analysis_v1 \
  --jobs protocol_manifests/causal_jobs_v2.csv \
  --audit analysis_outputs/causal_sweep_audit_v2 \
  --config config/config.yaml \
  --policy-python "$POLICY_PYTHON" \
  --rscript "$RSCRIPT" \
  --workers 2 \
  --output analysis_outputs/causal_policy_weights_v1
```

If this fails, preserve the partial output and diagnose the named immutable log.
Retry to `causal_policy_weights_v1_retry1`; never overwrite the first attempt.

## 5. Construct the 13 investable ensembles in target-weight space

```bash
"$PYTHON" -m publication_pipeline_draft.assemble_causal_policy_ensembles \
  --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
  --weight-manifest analysis_outputs/causal_policy_weights_v1/causal_policy_weight_manifest.csv \
  --repo-root . \
  --output analysis_outputs/causal_policy_ensembles_v1

(
  cd analysis_outputs/causal_policy_ensembles_v1
  sha256sum -c CONTENTS.sha256
)
```

The ensemble is the arithmetic mean of the ten target-weight vectors at each
decision date. Returns are never averaged. Turnover, financing, and realized
returns are recomputed from the ensemble target weights.

## 6. Materialize and execute common accounting

Use the exact realized asset panel from the completed v4 locked batch:

```bash
export REALIZED_PANEL=locked_evaluation/main_oos_v4_operational_retry/inputs/realized_asset_gross.csv

"$PYTHON" -m publication_pipeline_draft.materialize_causal_evaluation \
  --repo-root . \
  --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
  --analysis-release frozen_releases/causal_analysis_v1 \
  --policy-weights analysis_outputs/causal_policy_weights_v1 \
  --ensembles analysis_outputs/causal_policy_ensembles_v1 \
  --audit analysis_outputs/causal_sweep_audit_v2 \
  --realized "$REALIZED_PANEL" \
  --output analysis_outputs/causal_evaluation_interface_v1

"$PYTHON" -m publication_pipeline_draft.causal_common_accounting \
  --contract analysis_outputs/causal_evaluation_interface_v1/evaluation_contract.json \
  --realized "$REALIZED_PANEL" \
  --strategies analysis_outputs/causal_evaluation_interface_v1/strategy_manifest.csv \
  --output analysis_outputs/causal_common_accounting_v1
```

All 143 paths must join one-to-one to the same 24 realized holding periods and
asset order. The evaluator applies the same drifted-pretrade turnover, 10-bp
transaction cost, short-borrow cost, cash financing, and portfolio constraints.

## 7. Export the frozen causal panel and generate tables/plots

```bash
"$PYTHON" -m publication_pipeline_draft.export_causal_period_panel \
  --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
  --common-output analysis_outputs/causal_common_accounting_v1 \
  --output analysis_outputs/causal_strategy_periods_v1.csv

"$PYTHON" -m publication_pipeline_draft.analyze_causal_results \
  --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
  --period-panel analysis_outputs/causal_strategy_periods_v1.csv \
  --output analysis_outputs/causal_analysis_results_v1

(
  cd analysis_outputs/causal_analysis_results_v1
  sha256sum -c CONTENTS.sha256
)
```

The analysis emits six paper-ready CSV tables plus four plots in both 300-dpi
PNG and vector PDF form. The
primary table reports annualized CRRA certainty-equivalent effects, circular
moving-block-bootstrap intervals, raw p-values, Holm-adjusted p-values, and the
prospectively fixed decision label. The paired-seed tables are explicitly
labelled as training-randomness diagnostics only.

## 8. Interpretation boundary

Freeze the completed result package before manuscript work:

```bash
"$PYTHON" -m publication_pipeline_draft.freeze_causal_results \
  --repo-root . \
  --contract publication_pipeline_draft/config/causal_analysis_contract_v1.json \
  --analysis-release frozen_releases/causal_analysis_v1 \
  --evaluation-interface analysis_outputs/causal_evaluation_interface_v1 \
  --common-output analysis_outputs/causal_common_accounting_v1 \
  --period-panel analysis_outputs/causal_strategy_periods_v1.csv \
  --analysis-output analysis_outputs/causal_analysis_results_v1 \
  --output frozen_releases/causal_results_v1 \
  --archive frozen_releases/causal_results_v1.tar.gz

(
  cd frozen_releases/causal_results_v1
  sha256sum -c CONTENTS.sha256
)
```

Then interpret the frozen tables using these fixed rules:

- `component_supported`: positive CE effect and one-sided Holm-adjusted
  p-value at or below 0.05.
- `component_not_established`: the experiment does not establish a positive
  incremental contribution. This is not proof of equivalence.
- `opposite_direction_evidence`: the two-sided CE interval lies below zero.
- Algorithm comparisons are two-sided exploratory robustness checks.

Report every contrast regardless of sign. These results can explain the
consumed v4 result, but they cannot retroactively make that holdout confirmatory.
Independent evidence still requires the frozen walk-forward/external-market
program.
