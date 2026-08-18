# Focused retrospective walk-forward mechanism runbook

This is the compute-prioritized successor to the consumed causal analysis. It
uses the existing seven-asset history in two deterministic, non-overlapping
24-month windows. The design was chosen after the causal analysis and the panel
contains the consumed holdout, so it is retrospective robustness evidence—not
fresh confirmation. Its narrow purpose is to test the two decisive dependence
representations with 3 matched TD3 variants × 5 seeds per window.

## 1. Validate and freeze the prospective code

```bash
cd /gabirel/copula-portfolio-clean
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
export PYTHON=/gabirel/miniforge3/bin/python3
export RSCRIPT=/gabirel/miniforge3/bin/Rscript
export TRAIN_PYTHON=/gabirel/miniforge3/envs/vine-rl/bin/python
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python

bash hpc/validate_focused_walk_forward_v1.sh | \
  tee logs/focused_walk_forward_v1_validation.log

ENV_MANIFEST_DIR=protocol_manifests/focused_runtime_v1 \
EXPECTED_TRAIN_GPUS=4 \
CAPTURE_TIMING=prospective_before_focused_walk_forward_v1 \
bash hpc/capture_publication_environment.sh

"$PYTHON" -m publication_pipeline_draft.freeze_focused_walk_forward_release \
  --repo-root . \
  --runtime protocol_manifests/focused_runtime_v1 \
  --output frozen_releases/focused_walk_forward_mechanism_v1 \
  --archive frozen_releases/focused_walk_forward_mechanism_v1.tar.gz

(cd frozen_releases && \
  sha256sum -c focused_walk_forward_mechanism_v1.tar.gz.sha256)
```

## 2. Freeze the existing seven-asset panel and two windows

```bash
"$PYTHON" -m publication_pipeline_draft.focused_seven_asset_panel \
  --levels data/portfolio_B_7assets_2013.csv \
  --output frozen_releases/original_seven_asset_focused_development_v1

(cd frozen_releases/original_seven_asset_focused_development_v1 && \
  sha256sum -c CONTENTS.sha256)

"$PYTHON" -m publication_pipeline_draft.focused_walk_forward_windows \
  --monthly-panel frozen_releases/original_seven_asset_focused_development_v1/development_monthly_asset_gross.csv \
  --panel-manifest frozen_releases/original_seven_asset_focused_development_v1/development_panel_manifest.json \
  --output frozen_releases/retrospective_original_7asset_expanding_24m_v1_windows

(cd frozen_releases/retrospective_original_7asset_expanding_24m_v1_windows && \
  sha256sum -c CONTENTS.sha256)

"$PYTHON" -m publication_pipeline_draft.export_window_periods \
  --schedule frozen_releases/retrospective_original_7asset_expanding_24m_v1_windows/window_schedule.csv \
  --monthly-panel frozen_releases/original_seven_asset_focused_development_v1/development_monthly_asset_gross.csv \
  --output frozen_releases/retrospective_original_7asset_expanding_24m_v1_periods

column -s, -t < frozen_releases/retrospective_original_7asset_expanding_24m_v1_windows/window_schedule.csv
```

The schedule must contain exactly `w01` and `w02`. Do not shorten, slide, or
select windows after viewing returns.

## 3. Materialize and generate one window

Run the block first for `w01`, and later change only `WINDOW_ID` to `w02`.

```bash
export WINDOW_ID=retrospective_original_7asset_expanding_24m_v1_w01
export WINDOW_INPUT="frozen_releases/window_inputs/$WINDOW_ID"
export WINDOW_CONTRACT="frozen_releases/focused_window_contracts/$WINDOW_ID"
export WINDOW_ROOT="data/focused_original_7asset_runs_v1"

"$PYTHON" -m publication_pipeline_draft.materialize_window_return_input \
  --daily-returns frozen_releases/original_seven_asset_focused_development_v1/development_daily_log_returns.csv \
  --panel-manifest frozen_releases/original_seven_asset_focused_development_v1/development_panel_manifest.json \
  --schedule frozen_releases/retrospective_original_7asset_expanding_24m_v1_windows/window_schedule.csv \
  --window-id "$WINDOW_ID" \
  --reference-asset GOLD \
  --vine-truncation-level 0 \
  --output "$WINDOW_INPUT"

"$PYTHON" -m publication_pipeline_draft.focused_window_training_protocol \
  --repo-root . \
  --window-input "$WINDOW_INPUT" \
  --artifact-root "$WINDOW_ROOT" \
  --output "$WINDOW_CONTRACT"

(cd "$WINDOW_CONTRACT" && sha256sum -c CONTENTS.sha256)

"$PYTHON" -m publication_pipeline_draft.prepare_window_training_data \
  --repo-root . \
  --release frozen_releases/focused_walk_forward_mechanism_v1 \
  --contract "$WINDOW_CONTRACT" \
  --config config/config.yaml \
  --rscript "$RSCRIPT" \
  --sim-cores 60 \
  --log-root "logs/focused_generator_$WINDOW_ID"
```

The synthetic bundle is generated once per window and shared by all 15 matched
policies. Never regenerate it between experiments or edit a frozen release.

### Sampling-aware recovery for a completed strict-v1 generator

If simulation completed and wrote all artifacts but strict-v1 stopped only at
the diagnostic gate, do not select another simulation seed and do not rerun the
episodes. Preserve the failed bundle, revalidate the identical returns under
the sampling-aware guardrailed-v2 protocol, freeze a new operational release,
and adopt that attested bundle explicitly. This is a post-generation
statistical-protocol revision and cannot support a confirmatory claim.

```bash
export TRAINING_DIR="$WINDOW_ROOT/$WINDOW_ID/training_data"

mv "$TRAINING_DIR/synthetic_returns.RData" \
  "$TRAINING_DIR/synthetic_returns.strict_v1_failed.RData"
mv "$TRAINING_DIR/synthetic_bundle_manifest.json" \
  "$TRAINING_DIR/synthetic_bundle_manifest.strict_v1_failed.json"

"$RSCRIPT" --vanilla rl/revalidate_synthetic_bundle.r \
  "$TRAINING_DIR/synthetic_returns.strict_v1_failed.RData" \
  "$TRAINING_DIR/synthetic_returns.RData" \
  "$TRAINING_DIR/synthetic_bundle_manifest.json" \
  "$TRAINING_DIR/synthetic_diagnostics_sampling_v2" \
  "$TRAINING_DIR/synthetic_bundle_manifest.strict_v1_failed.json"
```

The revalidator reconstructs the unique historical training prefix from the
overlapping fine-tuning episodes, adds moving-block marginal intervals and
episode-clustered synthetic correlation intervals, retains the original strict
flags, and rewrites only metadata/diagnostic tables. The episode returns and
vine states are not regenerated. Complete the preparation step using a newly
frozen operational release and a newly materialized window contract:

```bash
"$PYTHON" -m publication_pipeline_draft.prepare_window_training_data \
  --repo-root . \
  --release frozen_releases/focused_walk_forward_mechanism_v2 \
  --contract "$WINDOW_CONTRACT" \
  --config config/config.yaml \
  --rscript "$RSCRIPT" \
  --sim-cores 60 \
  --adopt-existing-revalidated \
  --log-root "logs/focused_generator_${WINDOW_ID}_sampling_v2"

export FOCUSED_RELEASE=frozen_releases/focused_walk_forward_mechanism_v2
export WINDOW_CONTRACT="$WINDOW_CONTRACT_V2"
```

## 4. Train 15 policies on four A100s

The focused launcher reads `PRETRAIN_EPISODES` and `FINETUNE_EPISODES` from
the hash-attested window bundle manifest. These counts intentionally override
the original full-sample values in `config.yaml`. Verify this wiring without
creating outputs before every window sweep:

```bash
"$PYTHON" -m publication_pipeline_draft.run_focused_window_sweep \
  --repo-root . \
  --release "$FOCUSED_RELEASE" \
  --contract "$WINDOW_CONTRACT" \
  --config config/config.yaml \
  --train-python "$TRAIN_PYTHON" \
  --rscript "$RSCRIPT" \
  --gpus 0,1,2,3 \
  --cpu-cores 120 \
  --log-root "logs/focused_window_${WINDOW_ID}_planned" \
  --status "protocol_manifests/focused_window_${WINDOW_ID}_planned.csv" \
  --preflight-only
```

The reported episode counts must equal the attested bundle counts. A failed
operational attempt must be preserved under `failed_attempts/` and retried with
a new frozen source release, log root, and status path; it does not authorize
regenerating the synthetic bundle or changing the experiment matrix.

Early expanding windows can contain fewer overlapping historical trajectories
than the 24-step episode horizon. In that case a target-disjoint purged
validation episode is mathematically unavailable. Since this protocol fixes
the fine-tuning pass count at one, the trainer records
`fixed_one_pass_all_history_no_validation_short_window`, skips the unavailable
diagnostic selection run, and refits once on every attested historical
trajectory. It still fails closed if more than one selection pass is requested.

```bash
nohup "$PYTHON" -m publication_pipeline_draft.run_focused_window_sweep \
  --repo-root . \
  --release "${FOCUSED_RELEASE:-frozen_releases/focused_walk_forward_mechanism_v1}" \
  --contract "$WINDOW_CONTRACT" \
  --config config/config.yaml \
  --train-python "$TRAIN_PYTHON" \
  --rscript "$RSCRIPT" \
  --gpus 0,1,2,3 \
  --cpu-cores 120 \
  --log-root "logs/focused_window_$WINDOW_ID" \
  --status "protocol_manifests/focused_window_${WINDOW_ID}.csv" \
  > "logs/focused_window_${WINDOW_ID}.launch.log" 2>&1 &
echo $! > "logs/focused_window_${WINDOW_ID}.pid"

tail -f "logs/focused_window_${WINDOW_ID}.launch.log"
```

## 5. Audit and replay all 15 checkpoints

```bash
export WINDOW_STATUS="protocol_manifests/focused_window_${WINDOW_ID}.csv"
export WINDOW_AUDIT="analysis_outputs/focused_audit_${WINDOW_ID}"
export POLICY_WEIGHTS="publication_eval/focused_policy_weights_${WINDOW_ID}"

"$TRAIN_PYTHON" -m publication_pipeline_draft.audit_focused_window_sweep \
  --repo-root . \
  --contract "$WINDOW_CONTRACT" \
  --status "$WINDOW_STATUS" \
  --output "$WINDOW_AUDIT"

(cd "$WINDOW_AUDIT" && sha256sum -c CONTENTS.sha256)

"$PYTHON" -m publication_pipeline_draft.generate_focused_window_policy_weights \
  --repo-root . \
  --contract "$WINDOW_CONTRACT" \
  --audit "$WINDOW_AUDIT" \
  --config config/config.yaml \
  --policy-python "$POLICY_PYTHON" \
  --rscript "$RSCRIPT" \
  --workers 4 \
  --output "$POLICY_WEIGHTS"
```

Report-only economic warnings remain disclosed. Non-finite tensors, missing or
changed checkpoints, metadata mismatches, missing seeds, and hard-constraint
violations fail closed.

## 6. Generate the six financial benchmarks and score with common accounting

```bash
export PERIODS_FILE="frozen_releases/retrospective_original_7asset_expanding_24m_v1_periods/evaluation_periods_${WINDOW_ID}.csv"
export REALIZED_DIR="publication_eval/focused_realized_${WINDOW_ID}"
export FOCUSED_SCORE="publication_eval/focused_score_${WINDOW_ID}"
export FOCUSED_BENCHMARKS="publication_eval/focused_benchmarks_${WINDOW_ID}"

"$RSCRIPT" --vanilla \
  publication_pipeline_draft/build_window_realized_panel.R \
  "$WINDOW_INPUT/window_daily_log_returns.csv" \
  "$WINDOW_INPUT/return_input_manifest.json" \
  "$PERIODS_FILE" \
  "$WINDOW_ID" \
  "$REALIZED_DIR"

RETURNS_DATA_FILE="$WINDOW_INPUT/window_daily_log_returns.csv" \
RETURNS_DATA_KIND=daily_log_returns \
RETURNS_DATA_MANIFEST="$WINDOW_INPUT/return_input_manifest.json" \
EVAL_WINDOW_ID="$WINDOW_ID" \
TRAINING_MARGINALS_FILE="$WINDOW_ROOT/$WINDOW_ID/training_data/training_marginal_results.RData" \
NN_VINE_MODEL_DIR="$WINDOW_ROOT/$WINDOW_ID/training_data/nn_vine_models" \
BENCHMARK_METHODS=equal_weight,shrinkage_mean_variance,dcc_garch,static_vine,rolling_vine,dynamic_nn_vine \
"$RSCRIPT" --vanilla \
  publication_pipeline_draft/generate_benchmark_weights.R \
  config/config.yaml \
  publication_pipeline_draft/config/benchmark_contract_v3.json \
  "$FOCUSED_BENCHMARKS"

(cd "$FOCUSED_BENCHMARKS" && sha256sum weights_*.csv solver_audit.csv)

"$PYTHON" -m publication_pipeline_draft.score_focused_window \
  --inventory "$POLICY_WEIGHTS/focused_policy_weight_inventory.csv" \
  --audit "$WINDOW_AUDIT" \
  --realized "$REALIZED_DIR/realized_asset_gross.csv" \
  --benchmarks "$FOCUSED_BENCHMARKS" \
  --benchmark-contract publication_pipeline_draft/config/benchmark_contract_v3.json \
  --output "$FOCUSED_SCORE"

(cd "$FOCUSED_SCORE" && sha256sum -c CONTENTS.sha256)
```

Repeat steps 3–6 for `retrospective_original_7asset_expanding_24m_v1_w02`.

## 7. Combine the two windows and infer

```bash
"$PYTHON" -m publication_pipeline_draft.combine_focused_window_panels \
  --inputs \
    publication_eval/focused_score_retrospective_original_7asset_expanding_24m_v1_w01 \
    publication_eval/focused_score_retrospective_original_7asset_expanding_24m_v1_w02 \
  --output publication_eval/focused_walk_forward_period_panel_v1.csv

"$PYTHON" -m publication_pipeline_draft.analyze_focused_walk_forward \
  --protocol publication_pipeline_draft/config/focused_walk_forward_mechanisms_v1.json \
  --period-panel publication_eval/focused_walk_forward_period_panel_v1.csv \
  --output analysis_outputs/focused_walk_forward_mechanisms_v1

(cd analysis_outputs/focused_walk_forward_mechanisms_v1 && \
  sha256sum -c CONTENTS.sha256)
```

Inference uses a window-stratified moving-block bootstrap over time. Seeds are
optimization replicates and are never counted as independent market samples.
The two mechanism contrasts and the six compressed-CVaR-versus-financial-
benchmark comparisons are separate multiplicity families and are all reported.

## 8. Freeze the complete result while HPC storage is available

```bash
export W01=retrospective_original_7asset_expanding_24m_v1_w01
export W02=retrospective_original_7asset_expanding_24m_v1_w02

"$PYTHON" -m publication_pipeline_draft.freeze_focused_walk_forward_results \
  --prospective-release frozen_releases/focused_walk_forward_mechanism_v1 \
  --panel-release frozen_releases/original_seven_asset_focused_development_v1 \
  --window-release frozen_releases/retrospective_original_7asset_expanding_24m_v1_windows \
  --period-release frozen_releases/retrospective_original_7asset_expanding_24m_v1_periods \
  --contracts \
    "frozen_releases/focused_window_contracts/$W01" \
    "frozen_releases/focused_window_contracts/$W02" \
  --audits \
    "analysis_outputs/focused_audit_$W01" \
    "analysis_outputs/focused_audit_$W02" \
  --weights \
    "publication_eval/focused_policy_weights_$W01" \
    "publication_eval/focused_policy_weights_$W02" \
  --scores \
    "publication_eval/focused_score_$W01" \
    "publication_eval/focused_score_$W02" \
  --benchmarks \
    "publication_eval/focused_benchmarks_$W01" \
    "publication_eval/focused_benchmarks_$W02" \
  --combined-panel publication_eval/focused_walk_forward_period_panel_v1.csv \
  --analysis analysis_outputs/focused_walk_forward_mechanisms_v1 \
  --statuses \
    "protocol_manifests/focused_window_${W01}.csv" \
    "protocol_manifests/focused_window_${W02}.csv" \
  --output frozen_releases/focused_walk_forward_results_v1 \
  --archive frozen_releases/focused_walk_forward_results_v1.tar.gz

(cd frozen_releases && \
  sha256sum -c focused_walk_forward_results_v1.tar.gz.sha256)
(cd frozen_releases/focused_walk_forward_results_v1 && \
  sha256sum -c CONTENTS.sha256)
```

## Five-day HPC priority

1. Day 1: validate/freeze; generate and train `w01`.
2. Day 2: audit/replay/score `w01`; generate and train `w02`.
3. Day 3: audit/replay/score `w02`; combine and analyze.
4. Day 4: freeze archives, hashes, inventories, logs, and runtime evidence.
5. Day 5: reserve for operational retries and provenance gaps.

Do not add a post-selected third window. Defer the independent 18-ETF panel,
40-asset scalability study, and five-algorithm sweep until this package is
complete; each requires a distinct prospective contract.
