# Terminal robustness and reproducibility campaign v1

This campaign performs no training and no model selection. It snapshots six
completed evidence panels, reconstructs daily mark-to-market paths, re-scores
frozen weights over implementation-cost grids, runs registered resampling and
influence diagnostics, and then repeats the analysis as a clean-room byte-level
reproduction.

## Evidence authority

- Frozen primary evaluation remains the only frozen-primary evidence class.
- Causal, synthetic-dose, presentation-budget and matched-pretraining analyses
  remain post-holdout explanatory evidence.
- The two focused windows remain retrospective walk-forward evidence.
- Results must not be pooled across these classes to create a new confirmatory
  claim.
- The campaign's stop rule prohibits further same-holdout policy training or
  model selection.

## HPC execution

Run from the repository root with the established Python/R environments:

```bash
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
export PYTHON=/gabirel/miniforge3/bin/python3
export RSCRIPT=/gabirel/miniforge3/bin/Rscript
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python
export TERMINAL_WORKERS=76

bash hpc/run_terminal_robustness_v1.sh validate
bash hpc/run_terminal_robustness_v1.sh inputs
bash hpc/run_terminal_robustness_v1.sh freeze
bash hpc/run_terminal_robustness_v1.sh run
```

The run stage is asynchronous. Monitor without modifying its output:

```bash
bash hpc/run_terminal_robustness_v1.sh status
tail -f logs/terminal_robustness_v1.launch.log
```

After `terminal_robustness_campaign_complete` appears:

```bash
bash hpc/run_terminal_robustness_v1.sh verify
bash hpc/run_terminal_robustness_v1.sh cleanroom
bash hpc/run_terminal_robustness_v1.sh finalize
sha256sum -c terminal_robustness_v1_final.tar.gz.sha256
```

Do not delete the frozen release, reference results, or clean-room results until
the final archive and sidecar have both been copied off the HPC and independently
verified.

## Principal outputs

The final results directory contains:

- `daily_tail_risk_metrics.csv`;
- `daily_monthly_reconciliation.csv`;
- `friction_surface.csv`;
- `break_even_costs.csv`;
- `resampling_robustness.csv`;
- `leave_one_period_out.csv`;
- `white_reality_checks.csv`;
- `registered_contrast_robustness_summary.csv`;
- `primary_economic_metrics.csv`;
- `evidence_ledger.csv`.

Only the registered block-length-three moving-block result is the primary
resampling specification. Other block lengths and the stationary bootstrap are
sensitivity analyses. Daily 99% CVaR will usually contain fewer than twenty tail
events and must be labelled exploratory; the 95% daily tail metric is the main
daily tail-risk diagnostic.
