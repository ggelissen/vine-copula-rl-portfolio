# Frozen causal analysis decision

## Evidence status

The supplied causal-results archive and sidecar reconcile exactly at SHA-256
`ffcc34f0ab27777af5944f45c91faa487ef87fb6070e9df66f82a86fabae458e`.
Every top-level and nested `CONTENTS.sha256` inventory passes. The release is a
self-contained frozen result package with 130 audited policies, 13 arithmetic
target-weight ensembles, 24 locked periods per strategy, and 22 periods in the
declared complete-period analysis. It is post-holdout explanatory evidence and
cannot support a fresh confirmatory claim.

All checkpoint tensors are finite, metadata match the registered interventions,
the 70/31/29 operational carry-forward is disclosed, every portfolio constraint
passes, and every strategy is rescored with the same realized returns, drifted
turnover, transaction costs, and financing costs. Twenty-nine intended policies
are transparently retained under the registered report-only economic-behavior
rule; none failed tensor or hard-constraint checks.

## Scientific result

No preregistered positive component effect survives Holm correction. Six of
eight component contrasts are not established, while two point significantly
in the direction opposite to the proposed full-model contribution:

| Reference minus ablation | Annual CRRA CE effect | 95% block-bootstrap interval | Decision |
|---|---:|---:|---|
| Direct raw NN-vine state | -7.21% | [-13.68%, -2.60%] | Opposite-direction evidence |
| Scenario-CVaR observation | -5.65% | [-15.97%, 3.13%] | Not established |
| Joint policy-visible dependence | 2.50% | [-2.81%, 7.74%] | Not established |
| CVaR reward shaping | 0.84% | [-3.89%, 5.24%] | Not established |
| Synthetic NN-vine pretraining | -19.50% | [-37.06%, -6.46%] | Opposite-direction evidence |
| NN-vine generator versus temporal bootstrap | -0.60% | [-10.94%, 10.03%] | Not established |
| Recurrent encoder | -1.31% | [-5.03%, 1.98%] | Not established |
| Historical fine-tuning | -1.55% | [-9.91%, 5.41%] | Not established |

The full TD3 ensemble also has no established advantage over DDPG, SAC, PPO, or
A2C. All four algorithm intervals include zero and all Holm-adjusted p-values
are at least 0.932. Seeds are optimization replicates, not independent market
histories; sign instability across the ten matched seeds reinforces rather than
repairs the time-series uncertainty.

The full ensemble remains economically competent: 52.37% total return, 25.75%
CAGR, 12.30% annual volatility, 1.94 Sharpe, 5.94% maximum drawdown, and 24.87%
annual CRRA certainty equivalent over the 22 complete periods. It ranks 10th of
13 causal ensembles on CRRA certainty equivalent, but is second-lowest in
volatility and has the fourth-highest Sharpe. This is a risk-moderation result,
not component or algorithm superiority.

The historical-only ensemble is the best observed causal strategy (44.38%
annual CRRA CE and 100.69% total return), while the scenario-CVaR-only state is
second (32.08% annual CRRA CE and 69.41% total return). The historical-only
policies all triggered the registered turnover warning, and their ensemble
turnover is 0.614 per month versus 0.347 for the full model. This lowers their
behavioral confidence but does not erase the large after-cost result; the
common scorer already charges their higher transaction and financing costs.

## Interpretation and paper claim

The most plausible mechanism is state overload or redundant dependence
encoding. Conditional on the scalar scenario-CVaR signal, the high-dimensional
raw vine state degrades decisions. Removing both dependence channels is worse
than removing only raw vine features, which is consistent with useful
decision-aligned compression but is not statistically established. Synthetic
pretraining exhibits a material domain-gap signal relative to matched-update
historical training, although the historical control's turnover warning and
single consumed market path require independent validation.

The paper must not claim that the full NN-vine LSTM-TD3 architecture, synthetic
pretraining, recurrence, CVaR shaping, fine-tuning, or TD3 itself is proven
superior. A defensible result is that the frozen model is competitive and
risk-efficient, while the causal analysis identifies raw dependence-state
expansion and synthetic-to-historical transfer as the principal unresolved
failure modes.

## Compute decision

Do not rerun the 130-job causal study, tune against its holdout, or spend scarce
HPC time on more RL-algorithm controls. The next high-value computation is the
prospectively fixed focused retrospective walk-forward mechanism study:

1. full raw-vine state plus scenario-CVaR;
2. scenario-CVaR only (raw vine features zeroed);
3. no policy-visible vine dependence signal.

Use five matched seeds in the two deterministic non-overlapping 24-month
windows supported by the original seven-asset history. This costs 15 policies
per window instead of the broad framework's 50. It directly tests whether the
adverse raw-state effect and the conditional value of compressed scenario-CVaR
are stable across time. Because this reuses the same market history and was
designed after the causal result, it is retrospective robustness evidence, not
independent or confirmatory replication.
Historical-only pretraining is deliberately deferred until the window-specific
matched-update historical control has its own immutable generator and audit; it
must not be approximated by changing episode counts or reusing the seven-asset
bundle.

Cost sensitivity, ensemble-size sensitivity, tables, and plots can be computed
later from frozen weight paths on CPU. An independently sourced external-market
panel remains the proper later validation stage. The 40-asset scalability panel
and full five-algorithm sweep are lower priority than completing the two focused
seven-asset windows while GPU access remains available.
