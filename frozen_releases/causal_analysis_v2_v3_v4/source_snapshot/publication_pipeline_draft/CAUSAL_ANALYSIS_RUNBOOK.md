# Causal analysis v1: frozen three-revision explanatory workflow

The 13-by-10 training design is complete through the disclosed operational
sequence: 70 strict-gate v2 policies, 31 v3 policies whose strict gate passed,
and the exact 29 v3 failures retrained under the frozen v4 report-only branch.
No causal policy has yet been replayed on evaluation returns.

This study reuses the already consumed 24-month main holdout. It is therefore
post-holdout explanatory mechanism evidence, not a new confirmatory test. The
ten seeds quantify training randomness; they are not independent market paths.

## Runtime and immutable path contract

```bash
cd /gabirel/copula-portfolio-clean
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
export PYTHON=/gabirel/miniforge3/bin/python3
export TRAIN_PYTHON=/gabirel/miniforge3/envs/vine-rl/bin/python
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python
export RSCRIPT=/gabirel/miniforge3/bin/Rscript
export CAUSAL_REPLAY_WORKERS=4
```

The replay stage is CPU-bound; it does not train and does not need the A100s.
Four workers are conservative on the described 120-core host because each
evaluation process internally fixes vine simulation to one core.

The canonical staged driver is `hpc/finalize_causal_evaluation_v4.sh`. Every
stage refuses to overwrite an existing output. If an operational stage fails,
preserve its directory and diagnose it; use a versioned retry path rather than
deleting evidence or changing the scientific contract.

## 1. Preserve the completed v4 evidence

Before synchronization or evaluation, retain the three supplied files and
their hashes recorded in `V4_RETRY29_EVIDENCE.md`. The server-side source data
must also retain all v2, v3, and v4 run directories because the audit hashes the
exact 130 selected final checkpoints.

## 2. Validate current code

```bash
bash hpc/finalize_causal_evaluation_v4.sh preflight \
  | tee logs/causal_evaluation_v2_v3_v4_preflight.log
```

Expected: the Python suite passes, the causal contract validates, and the v4
release checksum passes. This happens before causal returns are read.

## 3. Merge the exact 70 + 31 + 29 training evidence

```bash
bash hpc/finalize_causal_evaluation_v4.sh merge \
  | tee logs/causal_evaluation_v2_v3_v4_merge.log
```

The merger verifies all three frozen releases, identical experiment/seed keys,
unchanged scientific settings, the diagnosed v3 trainer hash, the reviewed v4
trainer hash, and the exact 70/31/29 selection rule. It emits:

- `protocol_manifests/causal_jobs_v2_v3_v4_merged.csv` (130 jobs);
- `protocol_manifests/causal_sweep_status_v2_v3_v4_merged.csv` (130 passes);
- `protocol_manifests/causal_v2_v3_v4_operational_merge.json`.

## 4. Freeze the outcome-blind analysis plan

```bash
bash hpc/finalize_causal_evaluation_v4.sh freeze-plan \
  | tee logs/causal_evaluation_v2_v3_v4_freeze_plan.log
```

This binds the unchanged 12 contrasts and inference rules to the v2, v3, and
v4 training releases and the exact merge manifest. The freezer does not read
checkpoints, policy weights, realized returns, or causal outcomes.

## 5. Audit all 130 selected checkpoints

```bash
bash hpc/finalize_causal_evaluation_v4.sh audit \
  | tee logs/causal_evaluation_v2_v3_v4_audit.log
```

The GPU-training Python is used only because it can load every checkpoint. The
audit requires finite tensors, correct architecture/mode metadata, positive
update counters, exact 13-by-10 cardinality, and no hard exposure/position gate
failure. It preserves the true economic behavior diagnostic separately:
101 strict-path policies passed every diagnostic, while 29 report-only controls
remain eligible under the prospective intent-to-train rule. The immutable
audit directory is `analysis_outputs/causal_sweep_audit_v2_v3_v4`.

## 6. Replay the 130 policies and build 13 weight-space ensembles

This is the first causal-study stage that accesses the consumed holdout.

```bash
bash hpc/finalize_causal_evaluation_v4.sh replay \
  | tee logs/causal_evaluation_v2_v3_v4_replay.log

bash hpc/finalize_causal_evaluation_v4.sh ensembles \
  | tee logs/causal_evaluation_v2_v3_v4_ensembles.log
```

Every policy must export exactly 24 target-weight rows in the same asset order
and date order. Each experiment ensemble is the arithmetic mean of the ten
target-weight vectors at each date. Returns and costs are never averaged.

## 7. Apply common realized returns, constraints, and costs once

```bash
bash hpc/finalize_causal_evaluation_v4.sh accounting \
  | tee logs/causal_evaluation_v2_v3_v4_accounting.log
```

All 143 paths are joined one-to-one to
`locked_evaluation/main_oos_v4_operational_retry/inputs/realized_asset_gross.csv`.
The common scorer revalidates net/gross/position constraints and recomputes
drifted-pretrade turnover, 10-bp transaction cost, short-borrow cost, financing,
and net returns from target weights.

## 8. Run the preregistered analysis and freeze results

```bash
bash hpc/finalize_causal_evaluation_v4.sh analyze \
  | tee logs/causal_evaluation_v2_v3_v4_analysis.log

bash hpc/finalize_causal_evaluation_v4.sh freeze-results \
  | tee logs/causal_evaluation_v2_v3_v4_freeze_results.log
```

The analysis emits all 8 primary component contrasts, all 4 exploratory
algorithm contrasts, 9,999-replication circular moving-block intervals and
p-values, Holm adjustments within the declared families, seed-stability tables,
implementation diagnostics, and a separate table disclosing the 101/29
strict/report-only gate evidence. Paper figures are exported as 300-dpi PNG and
vector PDF.

The immutable deliverable is self-contained for result verification and
independent metric recomputation: it includes the frozen analysis plan, merged 130-job contract,
checkpoint audit, 130 policy weight logs, 13 ensemble weight logs, common
accounting panel, realized asset-return input, tables, and figures. Neural checkpoints are referenced by
their audited SHA-256 hashes rather than duplicated. The deliverable is:

```text
frozen_releases/causal_results_v2_v3_v4.tar.gz
frozen_releases/causal_results_v2_v3_v4.tar.gz.sha256
```

Copy both files, the merged manifest, checkpoint audit, and all stage logs off
the HPC before access expires. Preserve the exact selected checkpoints in
persistent storage if future policy replay must remain possible.

## 9. Fixed interpretation boundary

- `component_supported`: positive annualized CE effect and one-sided
  Holm-adjusted p-value at or below 0.05.
- `component_not_established`: positive contribution was not established; this
  is not proof of equivalence.
- `opposite_direction_evidence`: the two-sided CE interval lies below zero.
- Algorithm comparisons are two-sided exploratory robustness checks.
- Report every contrast regardless of sign.
- These results may explain the consumed main result, but cannot retroactively
  make it confirmatory. Independent evidence still requires the frozen
  walk-forward/external-market program.
