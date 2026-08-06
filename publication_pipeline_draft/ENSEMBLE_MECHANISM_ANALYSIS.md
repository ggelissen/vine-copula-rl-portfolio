# Post-holdout ensemble-mechanism analysis

`analyze_ensemble_mechanism.py` is a separate, fail-closed explanatory layer
for an already completed locked evaluation batch. It does not modify or call
the primary evaluator, change the frozen candidate, or create a new
confirmatory test.

## Classification

Every artifact is labelled:

> POST-HOLDOUT EXPLANATORY — NOT CONFIRMATORY

This analysis may explain why a predeclared arithmetic seed ensemble behaves
differently from its individual policies. It must not be used to select a
favourable seed count, seed subset, turnover convention, or revised primary
estimand after seeing the holdout.

## Fail-closed inputs

The CLI accepts only a completed locked-batch directory. Before producing an
output it verifies:

1. `locked_batch_manifest.json` reports `status=complete`, confirms holdout
   access, and records the contractually expected number of full policies.
2. The publication manifest names the requested arithmetic ensemble as the
   frozen primary strategy.
3. Every seed has a unique numerical seed, weight path, weight SHA-256 and
   checkpoint SHA-256.
4. Every frozen weight file hashes to the manifest value and matches its
   scored-panel weights.
5. Every strategy shares the same locked period keys.
6. Ensemble weights equal the date-by-date arithmetic mean of every seed
   weight, within the declared tolerance.
7. Ensemble gross returns equal the date-by-date mean seed gross return.
8. Target-to-target rescoring reproduces every frozen seed and ensemble net
   return before drift-aware sensitivity is calculated.

The output directory must not exist. Results are written through a temporary
directory and atomically installed only after successful completion.

## Outputs

The explanatory tables cover:

- recomputed per-seed and ensemble performance;
- seed dispersion and benchmark win fractions;
- ensemble-versus-seed ranks;
- pairwise return and flattened-weight correlations;
- sign disagreement and cross-seed weight dispersion;
- gross, short, turnover, trading-cost and financing-cost cancellation;
- exact gross-return averaging versus net cost savings;
- deterministic sorted-prefix and seed-bootstrap k-seed sensitivity;
- target-to-target versus realized-drift-aware turnover sensitivity;
- frozen input paths, hashes and ensemble-identity tolerances.

Figures are emitted as PNG and PDF and carry the post-holdout warning in the
image itself.

## k-seed sensitivity

Two deliberately exploratory paths are reported:

- **Sorted prefix:** the first `k` seeds in numerical order. This is
  deterministic and auditable, but the ordering is arbitrary.
- **Seed bootstrap:** `k` seeds sampled with replacement from the frozen seed
  population using the analysis contract's fixed random seed. This describes
  sensitivity to algorithm-seed composition; it does not resample market
  months and is not a market-performance confidence interval.

No k is promoted as optimal.

## Drift-aware turnover sensitivity

The frozen evaluator uses full-L1 target-to-target turnover. The explanatory
sensitivity preserves the same full-L1 definition and cost rate, but updates
the previous portfolio after realized asset drift:

\[
  w^{\mathrm{drift}}_{i,t}
  = \frac{w_{i,t-1}g_{i,t-1}}
  {1 + w_{t-1}^{\top}(g_{t-1}-1)}.
\]

The sensitivity is calculated through all locked periods before the declared
reporting scope is applied, so an excluded shortened period does not break the
portfolio transition path. It is not a replacement primary cost convention.

## Run

```powershell
python publication_pipeline_draft/analyze_ensemble_mechanism.py `
  --batch analysis_outputs/oos_v4_verified_770d2944/main_oos_v4_operational_retry `
  --contract publication_pipeline_draft/config/ensemble_mechanism_contract.example.json `
  --output analysis_outputs/oos_v4_verified_770d2944/post_holdout_ensemble_mechanism_v1
```

The example contract is appropriate for the verified 20-seed v4 batch. Copy
it to a frozen analysis-specific path if any explanatory setting must be
changed; never silently overwrite a completed output.
