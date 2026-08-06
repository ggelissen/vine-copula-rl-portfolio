# Post-sweep integration runbook

Do not inspect or execute the final holdout merely because the training sweep
finishes. The benchmark algorithms and the comparison contract must be frozen
first.

## Phase A: still no holdout access

1. Confirm `data/rl_runs/seed_sweep_status.csv` contains exactly the 20
   preregistered seeds, all training and sanity exit codes are zero, and every
   no-holdout gate passes.
2. Run `diagnostic_artifacts.py --rl-runs ... --expected-seeds 20` and inspect
   training stability. This uses held-in/synthetic diagnostics only.
3. Freeze all seed directories, checkpoints, training data hashes, config, and
   source files without opening the holdout:

   ```bash
   REPO_ROOT="$(pwd -P)"
   DIAGNOSTIC_DIR="$REPO_ROOT/data/publication_training_artifacts_20seeds"
   DIAGNOSTIC_ARCHIVE="$REPO_ROOT/training_artifacts_20seeds.tar.gz"

   test -d "$DIAGNOSTIC_DIR" || { echo "Missing $DIAGNOSTIC_DIR" >&2; exit 1; }
   tar -C "$(dirname "$DIAGNOSTIC_DIR")" \
     -czf "$DIAGNOSTIC_ARCHIVE" "$(basename "$DIAGNOSTIC_DIR")"
   test -s "$DIAGNOSTIC_ARCHIVE" || { echo "Archive was not created" >&2; exit 1; }

   python publication_pipeline_draft/freeze_training_release.py \
     --repo-root "$REPO_ROOT" \
     --rl-runs "$REPO_ROOT/data/rl_runs" \
     --diagnostics-archive "$DIAGNOSTIC_ARCHIVE" \
     --expected-seeds 20 \
     --output "$REPO_ROOT/frozen_releases/training_schema5_v1" \
     --bundle "$REPO_ROOT/frozen_releases/training_schema5_v1.tar.gz"

   (cd "$REPO_ROOT/frozen_releases" && \
     sha256sum -c training_schema5_v1.tar.gz.sha256)
   ```

   The command must run in the checkout used on the training machine. A local
   clone with different byte hashes is not a substitute. Copy the bundle and
   its `.sha256` sidecar to immutable storage.
4. Merge the stricter common-evaluator validation into the live code only now
   that no later seed can source a changed file.
5. Implement every main-table benchmark in `BENCHMARK_SPECIFICATION.md` as a
   causal weight generator. Run its causality, constraint, reproducibility,
   and solver-failure tests on development dates only.
6. Leave incomplete ablation/sensitivity tables explicitly absent from the
   confirmatory main release. Never infer missing rows from the full model.
7. Freeze and hash:
   - one source snapshot/tag;
   - one master configuration;
   - the explicit training cutoff and 24 holding-period keys;
   - every strategy and inference pair;
   - the complete-period primary scope and shortened-period robustness scope;
   - cost, risk-free-rate, block-length, bootstrap seed, and replication count.

## Phase B: one locked batch

1. Export the realized panel without viewing strategy performance:

   ```bash
   Rscript --vanilla publication_pipeline_draft/build_realized_panel.R \
     config/config.yaml publication_eval/inputs
   ```

2. Generate all 20 full-policy weight logs and every main benchmark weight log
   in one scripted batch. Secondary ablations enter only a separately declared
   release when their frozen artifacts already exist.
3. Verify all logs exist, contain exactly the locked keys, and hash them. Fill
   a copy of `config/strategy_manifest.example.csv`. Only gate-passing seeds
   enter the predeclared ensemble; the expected contract requires 20.
4. Run the common evaluator once into a new, non-existing directory:

   ```bash
   python publication_pipeline_draft/publication_pipeline.py \
     --contract publication_eval/evaluation_contract.json \
     --realized publication_eval/inputs/realized_asset_gross.csv \
     --strategies publication_eval/strategy_manifest.csv \
     --output publication_eval/results_locked_oos_v1
   ```

5. Archive the entire output directory before looking at tables or figures.
   Technical reruns must use identical hashes and a new directory; analytical
   changes constitute a new preregistered experiment, not a correction to the
   first holdout result.

## Phase C: interpretation

- Lead with economic effect sizes and intervals, not a win/loss count.
- Treat the 20 seeds as optimisation variability. Only the investable ensemble
  enters primary market-path inference.
- State that the complete-period primary scope has fewer than 24 observations;
  show `locked_all` as a disclosed robustness result including the shortened
  terminal interval.
- Label empirical monthly 5% CVaR as descriptive and report its event count.
- A superiority claim requires the proposed ensemble to compare favorably with
  the relevant benchmark family after common costs and Holm adjustment, not
  merely to beat equal weight in one Sharpe estimate.
