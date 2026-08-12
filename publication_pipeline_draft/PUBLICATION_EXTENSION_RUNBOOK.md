# Publication Extension v2 Runbook

> The first full v2 training attempt completed 70/130 jobs.  For the disclosed
> post-diagnostic revision and all new execution, follow
> `PUBLICATION_EXTENSION_V3_RECOVERY.md`; the commands below remain the frozen
> historical v2 protocol and must not be reused under the v3 label.

This framework is prospective. It does not modify the consumed 24-month main
evaluation or the frozen post-holdout no-vine ablation. Manuscript editing stays
blocked until the resulting artifacts have passed the gates below.

## Evidence boundaries

- The v4 main holdout is consumed evidence.
- `post_holdout_explanatory_ablation_v2` is frozen explanatory evidence; it is
  not a new confirmatory test.
- Retrospective walk-forward results are robustness/development evidence.
- A new asset panel over an already observed calendar supplies cross-sectional
  external validity, not temporal independence.
- Only data after the consumed holdout, committed through the separate access
  ledger before first access, can support a fresh confirmatory claim.

## A. Verify the registered ablation

```bash
cd /gabirel/copula-portfolio-clean
sha256sum -c \
  frozen_releases/post_holdout_explanatory_ablation_v2/post_holdout_explanatory_ablation_v2_operational_retry.tar.gz.sha256
python3 -m json.tool \
  frozen_releases/post_holdout_explanatory_ablation_v2/registration_manifest.json
```

The expected archive digest is
`8457cf6fca34526591ac89d356cf52ca99c69a7789035a41dfbc41345f8595ff`.

## B. Validate code before any new experiment

```bash
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
export PYTHON=/gabirel/miniforge3/bin/python3
export RSCRIPT=/gabirel/miniforge3/bin/Rscript
export TRAIN_PYTHON=/gabirel/miniforge3/envs/vine-rl/bin/python
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python
bash hpc/validate_publication_extension_v2.sh
```

The Python environment used by the R trainer must additionally import its
pinned CUDA PyTorch and `gymnasium`. Do not install packages during a locked
evaluation.

## C. Materialize the development job contracts

```bash
mkdir -p protocol_manifests
python3 publication_pipeline_draft/publication_research_program.py validate \
  --program publication_pipeline_draft/config/publication_research_program_v2.json
python3 publication_pipeline_draft/publication_research_program.py jobs \
  --program publication_pipeline_draft/config/publication_research_program_v2.json \
  --output protocol_manifests/publication_extension_jobs_v2.csv
python3 publication_pipeline_draft/causal_ablation_protocol.py \
  --output-root data/publication_extension_runs_v2 \
  --output protocol_manifests/causal_jobs_v2.csv
```

The research-program matrix has 560 design-level rows. The causal training
matrix has 130 jobs: 13 experiments times 10 identical seeds.

## D. Generate the two alternative pretraining controls

Run once. The command refuses to overwrite existing bundles.

```bash
Rscript --vanilla rl/generate_ablation_training_bundles.r \
  data/synthetic_returns.RData \
  data/ablation_training_bundles \
  config/config.yaml
(cd data/ablation_training_bundles && sha256sum -c CONTENTS.sha256)
```

Never regenerate a bundle merely because its hash is inconvenient to verify.

## E. Development smoke tests

Do not create a shortened training bundle: that would no longer test the
all-synthetic-data training contract, and too few episodes can turn the
behavior gate into a misleading failure. The validation script instead runs
unit/protocol checks, CUDA construction tests for every modern RL control, and
the isolated inference smoke test before GPU-hours are spent. Confirm:

- every checkpoint records schema, algorithm, encoder, all three signal modes,
  pretraining source, and fine-tuning status;
- DDPG, SAC, PPO, A2C and feedforward TD3 can save, reload, and infer through
  `policy_inference_server_v2.py`;
- every environment action satisfies net, gross and position limits;
- PPO uses episodic GAE/PPO updates rather than the off-policy replay buffer;
- failed training leaves a failed row and is not replaced by another seed.

## F. Full causal sweep (later execution step)

After smoke-test corrections are finished, rerun every test, regenerate the
final job matrix in a new path, capture the actual runtime, and freeze the
pre-execution extension release:

```bash
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python
export TRAIN_PYTHON=/gabirel/miniforge3/envs/vine-rl/bin/python
export RSCRIPT=/gabirel/miniforge3/bin/Rscript
export PYTHON=/gabirel/miniforge3/bin/python3
export EXPECTED_TRAIN_GPUS=4
export CAPTURE_TIMING=prospective_before_publication_extension_v2
export ENV_MANIFEST_DIR="$PWD/provenance_environment_extension_v2"
bash hpc/capture_publication_environment.sh

python3 publication_pipeline_draft/freeze_publication_extension.py \
  --repo-root . \
  --jobs protocol_manifests/causal_jobs_v2.csv \
  --runtime provenance_environment_extension_v2 \
  --bundle-manifest data/ablation_training_bundles/ablation_bundle_manifest.csv \
  --output frozen_releases/publication_extension_v2 \
  --archive frozen_releases/publication_extension_v2.tar.gz
```

Verify both `CONTENTS.sha256` and the archive sidecar before training. The
following sweep must use the source snapshot/contract represented by that
freeze; operational corrections require a versioned v3 release.

Set the actual pinned executables first. Example for four GPUs:

```bash
export TRAIN_PYTHON=/gabirel/miniforge3/envs/vine-rl/bin/python
export RSCRIPT=/gabirel/miniforge3/bin/Rscript

nohup python3 publication_pipeline_draft/run_causal_sweep.py \
  --jobs protocol_manifests/causal_jobs_v2.csv \
  --release frozen_releases/publication_extension_v2 \
  --repo-root . \
  --config config/config.yaml \
  --train-python "$TRAIN_PYTHON" \
  --rscript "$RSCRIPT" \
  --gpus 0,1,2,3 \
  --cpu-cores 100 \
  --log-root logs/publication_extension_v2 \
  --status protocol_manifests/causal_sweep_status_v2.csv \
  > logs/publication_extension_v2.launch.log 2>&1 &
```

Use `--experiments id1,id2` only for a preregistered shard; merge shards by an
audited script before the checkpoint audit. Example after all disjoint shards
finish:

```bash
python3 publication_pipeline_draft/merge_causal_sweep_status.py \
  --jobs protocol_manifests/causal_jobs_v2.csv \
  --statuses protocol_manifests/causal_status_shard_*.csv \
  --output protocol_manifests/causal_sweep_status_v2.csv
```

Do not relaunch a failed seed into a new
directory without preserving and explaining the first failure.

## G. Audit and freeze (later execution step)

```bash
"$TRAIN_PYTHON" publication_pipeline_draft/audit_causal_sweep.py \
  --jobs protocol_manifests/causal_jobs_v2.csv \
  --status protocol_manifests/causal_sweep_status_v2.csv \
  --repo-root . \
  --output analysis_outputs/causal_sweep_audit_v2
```

All ten seeds for all thirteen experiments must pass. The auditor verifies tensor
finiteness, behavioral gates, exact checkpoint metadata, and the feedforward
parameter-count tolerance. Freeze code, runtime, data hashes, job matrix,
checkpoints and audit before any external/future test is opened.

## H. External panel and walk-forward development

Fill `external_panel_metadata.example.json` before accessing returns. The main
panel is fixed at 18 liquid USD total-return ETFs. Use licensed adjusted
total-return levels; do not silently substitute unadjusted close.

```bash
python3 publication_pipeline_draft/asset_panel_protocol.py \
  --levels /licensed/input/global_liquid_etf_18_levels.csv \
  --metadata /licensed/input/global_liquid_etf_18_metadata.json \
  --validation-end 2026-07-06 \
  --earliest-future-test-start 2026-07-07 \
  --output frozen_releases/global_liquid_etf_18_development

python3 publication_pipeline_draft/walk_forward_windows.py \
  --design-id retrospective_expanding_24m_v1 \
  --monthly-panel frozen_releases/global_liquid_etf_18_development/development_monthly_asset_gross.csv \
  --panel-manifest frozen_releases/global_liquid_etf_18_development/development_panel_manifest.json \
  --output frozen_releases/retrospective_expanding_24m_v1_windows

python3 publication_pipeline_draft/export_window_periods.py \
  --schedule frozen_releases/retrospective_expanding_24m_v1_windows/window_schedule.csv \
  --monthly-panel frozen_releases/global_liquid_etf_18_development/development_monthly_asset_gross.csv \
  --output frozen_releases/retrospective_expanding_24m_v1_periods
```

Insufficient history is a protocol failure, not permission to shorten windows
after seeing results.

### H1. Freeze a dimension-matched return input for each window

The original seven-asset checkpoint is never reused on an 18- or 40-asset
panel. Each window is truncated at its own test end; the generator/trainer then
reserve its final 24 periods exactly. Example for window 1:

```bash
WINDOW_ID=retrospective_expanding_24m_v1_w01
WINDOW_INPUT=frozen_releases/window_inputs/$WINDOW_ID

python3 publication_pipeline_draft/materialize_window_return_input.py \
  --daily-returns frozen_releases/global_liquid_etf_18_development/development_daily_log_returns.csv \
  --panel-manifest frozen_releases/global_liquid_etf_18_development/development_panel_manifest.json \
  --schedule frozen_releases/retrospective_expanding_24m_v1_windows/window_schedule.csv \
  --window-id "$WINDOW_ID" \
  --reference-asset BIL \
  --vine-truncation-level 0 \
  --output "$WINDOW_INPUT"

(cd "$WINDOW_INPUT" && sha256sum -c CONTENTS.sha256)
```

For the separately frozen 40-asset scalability panel, set
`--vine-truncation-level 3`. A zero level means all `d-1` trees and is reserved
for the 18-asset main external panel. The deterministic order search uses exact
enumeration only through nine assets and an all-start greedy plus 2-opt search
above that, avoiding the old factorial runtime.

### H2. Freeze the per-window generator and 50-policy comparison contract

```bash
WINDOW_CONTRACT=frozen_releases/window_training_contracts/$WINDOW_ID

python3 publication_pipeline_draft/window_training_protocol.py \
  --repo-root . \
  --window-input "$WINDOW_INPUT" \
  --artifact-root data/external_window_runs_v2 \
  --output "$WINDOW_CONTRACT"

(cd "$WINDOW_CONTRACT" && sha256sum -c CONTENTS.sha256)
python3 -m json.tool "$WINDOW_CONTRACT/window_training_manifest.json"
```

The contract contains TD3, DDPG, SAC, PPO and A2C under the same LSTM,
observations, economic reward, constraints, costs and ten matched seeds.
Algorithm-specific optimizer defaults are fixed prospectively; post-window
tuning is forbidden.

### H3. Generate and diagnose the window's training bundle once

```bash
python3 publication_pipeline_draft/prepare_window_training_data.py \
  --repo-root . \
  --release frozen_releases/publication_extension_v2 \
  --contract "$WINDOW_CONTRACT" \
  --config config/config.yaml \
  --rscript "$RSCRIPT" \
  --sim-cores 24 \
  --log-root "logs/window_generator_$WINDOW_ID"
```

This step fails unless all marginal, correlation, lower-tail and temporal gates
pass. The single generated bundle and historical fine-tuning prefix are shared
across algorithms; only policy-optimization seeds vary.

### H4. Run the matched five-algorithm sweep (later execution)

```bash
nohup python3 publication_pipeline_draft/run_window_rl_sweep.py \
  --repo-root . \
  --release frozen_releases/publication_extension_v2 \
  --contract "$WINDOW_CONTRACT" \
  --config config/config.yaml \
  --train-python "$TRAIN_PYTHON" \
  --rscript "$RSCRIPT" \
  --gpus 0,1,2,3 \
  --cpu-cores 100 \
  --log-root "logs/window_rl_$WINDOW_ID" \
  --status "protocol_manifests/window_rl_${WINDOW_ID}.csv" \
  > "logs/window_rl_${WINDOW_ID}.launch.log" 2>&1 &
```

The runner also executes the no-holdout sanity test. A run passes only when
training, checkpoint creation, behavior gates and sanity checks all pass; no
failed seed is replaced.

### H5. Audit all 50 checkpoints, then export all 50 policy logs

Use the CUDA training Python for tensor inspection and the isolated CPU policy
Python for inference. The audit rejects non-finite tensors, metadata drift,
partial seed success, or failed behavior/sanity evidence.

```bash
WINDOW_STATUS="protocol_manifests/window_rl_${WINDOW_ID}.csv"
WINDOW_AUDIT="analysis_outputs/window_rl_audit_${WINDOW_ID}"
POLICY_WEIGHTS="publication_eval/window_policy_weights_${WINDOW_ID}"

"$TRAIN_PYTHON" publication_pipeline_draft/audit_window_rl_sweep.py \
  --repo-root . \
  --release frozen_releases/publication_extension_v2 \
  --contract "$WINDOW_CONTRACT" \
  --status "$WINDOW_STATUS" \
  --output "$WINDOW_AUDIT"

python3 publication_pipeline_draft/generate_window_policy_weights.py \
  --repo-root . \
  --release frozen_releases/publication_extension_v2 \
  --contract "$WINDOW_CONTRACT" \
  --sweep-status "$WINDOW_STATUS" \
  --policy-python "$POLICY_PYTHON" \
  --rscript "$RSCRIPT" \
  --workers 4 \
  --output "$POLICY_WEIGHTS"
```

Do not evaluate only the seed ensemble. The common evaluator retains all 50
individual policies for seed-distribution reporting and constructs each
ten-seed ensemble from preregistered arithmetic mean target weights.

## I. Extended financial benchmarks

For each frozen window calendar generated above, run the benchmark generator;
the example below uses window 1:

```bash
Rscript --vanilla publication_pipeline_draft/generate_extended_benchmark_weights.R \
  "$WINDOW_INPUT/window_daily_log_returns.csv" \
  frozen_releases/retrospective_expanding_24m_v1_periods/evaluation_periods_retrospective_expanding_24m_v1_w01.csv \
  publication_pipeline_draft/config/benchmark_contract_v2.json \
  "publication_eval/extended_benchmarks_$WINDOW_ID" \
  "$WINDOW_INPUT/return_input_manifest.json"
```

The generator fails on future-data access, calendar mismatch, constraint
failure, or non-converged optimization. It never silently falls back.

Generate the six existing benchmarks from the same input and panel-specific
vine artifacts (the environment variables override the seven-asset defaults):

```bash
export RETURNS_DATA_FILE="$WINDOW_INPUT/window_daily_log_returns.csv"
export RETURNS_DATA_KIND=daily_log_returns
export RETURNS_DATA_MANIFEST="$WINDOW_INPUT/return_input_manifest.json"
export REF_COL=18
export VINE_TRUNCATION_LEVEL=17
export TRAINING_MARGINALS_FILE="data/external_window_runs_v2/$WINDOW_ID/training_data/training_marginal_results.RData"
export NN_VINE_MODEL_DIR="data/external_window_runs_v2/$WINDOW_ID/training_data/nn_vine_models"
export EVAL_WINDOW_ID="$WINDOW_ID"

Rscript --vanilla publication_pipeline_draft/generate_benchmark_weights.R \
  config/config.yaml \
  publication_pipeline_draft/config/benchmark_contract_v2.json \
  "publication_eval/core_benchmarks_$WINDOW_ID"
```

The complete comparison now contains 11 financial benchmarks: equal weight,
minimum variance, long-only risk parity, shrinkage mean-variance, empirical
mean-CVaR, 12-1 momentum, Black-Litterman momentum views, DCC-GARCH, static
vine, rolling vine, and dynamic NN-vine without RL. Every numerical optimizer
must have an accepted convergence code; no fallback portfolio is permitted.

## J. Build and execute the common per-window evaluation

First build the common realized panel from exactly the frozen daily return
input and canonical 24-period calendar. Then assemble the evaluation protocol.
The assembler verifies all solver audits, all checkpoint hashes, exact policy
membership, dates, and asset order before it emits an immutable contract.

```bash
PERIODS_FILE="frozen_releases/retrospective_expanding_24m_v1_periods/evaluation_periods_${WINDOW_ID}.csv"
REALIZED_DIR="publication_eval/realized_${WINDOW_ID}"
EVAL_PROTOCOL="frozen_releases/window_evaluation_contracts/${WINDOW_ID}_v1"
EVAL_OUTPUT="publication_eval/results_${WINDOW_ID}_v1"

"$RSCRIPT" --vanilla \
  publication_pipeline_draft/build_window_realized_panel.R \
  "$WINDOW_INPUT/window_daily_log_returns.csv" \
  "$WINDOW_INPUT/return_input_manifest.json" \
  "$PERIODS_FILE" \
  "$WINDOW_ID" \
  "$REALIZED_DIR"

python3 publication_pipeline_draft/window_evaluation_protocol.py \
  --repo-root . \
  --release frozen_releases/publication_extension_v2 \
  --window-contract "$WINDOW_CONTRACT" \
  --checkpoint-audit "$WINDOW_AUDIT" \
  --policy-inventory "$POLICY_WEIGHTS/policy_weight_inventory.csv" \
  --core-benchmarks "publication_eval/core_benchmarks_$WINDOW_ID" \
  --extended-benchmarks "publication_eval/extended_benchmarks_$WINDOW_ID" \
  --realized-panel "$REALIZED_DIR/realized_asset_gross.csv" \
  --output "$EVAL_PROTOCOL"

(cd "$EVAL_PROTOCOL" && sha256sum -c CONTENTS.sha256)

python3 publication_pipeline_draft/execute_window_evaluation.py \
  --protocol "$EVAL_PROTOCOL" \
  --realized "$REALIZED_DIR/realized_asset_gross.csv" \
  --daily-returns "$WINDOW_INPUT/window_daily_log_returns.csv" \
  --return-manifest "$WINDOW_INPUT/return_input_manifest.json" \
  --output "$EVAL_OUTPUT" \
  --bundle "${EVAL_OUTPUT}.tar.gz"

(cd "$(dirname "$EVAL_OUTPUT")" && \
  sha256sum -c "$(basename "$EVAL_OUTPUT").tar.gz.sha256")
python3 -m json.tool \
  "$EVAL_OUTPUT/window_evaluation_execution_manifest.json"
```

The common monthly evaluator produces return/risk/CRRA tables, paired HAC and
moving-block inference, Holm corrections, White's reality check, constraint
audits, fixed-target cost/borrow grids, and exhaustive ensemble-size
sensitivity over every seed subset for k = 1, 2, 3, 5, 10. The separate daily
mark-to-market audit reports daily volatility, downside deviation, VaR, CVaR,
tail counts and drawdown and must reconcile exactly to monthly common-path
accounting.

## K. Aggregate non-overlapping retrospective windows

Only after at least two complete non-overlapping per-window evaluations exist:

```bash
python3 publication_pipeline_draft/aggregate_walk_forward_results.py \
  --results \
    publication_eval/results_retrospective_expanding_24m_v1_w01_v1/common_evaluator \
    publication_eval/results_retrospective_expanding_24m_v1_w02_v1/common_evaluator \
    publication_eval/results_retrospective_expanding_24m_v1_w03_v1/common_evaluator \
    publication_eval/results_retrospective_expanding_24m_v1_w04_v1/common_evaluator \
  --output analysis_outputs/retrospective_walk_forward_v1
```

This uses a window-stratified circular moving-block bootstrap. Its output is
development/robustness evidence, not a replacement confirmatory holdout.

## L. Forty-asset scalability panel

The universe is now fixed prospectively in
`config/scalability_universe_v1.json`: 40 liquid cross-asset ETFs, reference
asset BIL, and a three-tree truncated D-vine. Populate the remaining vendor,
license, retrieval, inception/delisting and date fields in
`scalability_panel_metadata.example.json`; do not change the symbol order after
viewing returns. This panel addresses computational and cross-sectional scale,
but its explicit surviving-ETF selection limitation must remain in the paper.

## M. Fresh temporal confirmation

Do not use the development window generator for future observations. Use
`future_confirmatory_protocol.py` with a data custodian, a pre-access artifact
manifest, container/runtime digest, frozen model selection, and the immutable
access ledger. At least two non-overlapping future windows are required by the
current program before claiming temporal confirmation.
