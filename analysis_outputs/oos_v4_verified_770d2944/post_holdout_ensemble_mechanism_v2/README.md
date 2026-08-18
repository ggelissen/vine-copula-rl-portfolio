# Post-holdout explanatory ensemble-mechanism analysis

**POST-HOLDOUT EXPLANATORY — NOT CONFIRMATORY**

These outputs explain a frozen, predeclared arithmetic seed ensemble after
holdout access. They do not amend the locked evaluation, select a new primary
strategy, or support new confirmatory p-values.

- Seed count required: 20
- Frozen ensemble: `vine_td3_ensemble`
- Reported scope: `complete_periods`
- k-seed results: deterministic sorted prefixes and a deterministic
  seed-bootstrap with replacement. Both are exploratory because they were
  specified after holdout access.
- Drift-aware turnover: recomputes the previous portfolio after realized asset
  drift while preserving the evaluator's full-L1 convention and economic cost
  rates. It is sensitivity analysis, not a replacement primary result.

All CSV files include `analysis_classification` and
`confirmatory_use_permitted` columns. Every figure carries the same warning.
