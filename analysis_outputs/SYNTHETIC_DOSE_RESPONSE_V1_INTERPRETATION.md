# Synthetic-dose response v1: evidence interpretation

## Evidence status and integrity

The archive `synthetic_dose_response_v1_final.tar.gz` has SHA-256
`ed6dfbf4a17721081a296b08647a9eff02fa19f3fc8f890069d973b3cde4ada4`,
matching its supplied checksum. The frozen release records 20 policies, two
architectures, ten matched optimization seeds, 100 unique synthetic episodes,
100 synthetic episode presentations, and 61 historical fine-tuning episodes.
All 20 jobs exited successfully. All checkpoints were finite, matched their
declared architecture, and passed the registered economic behaviour checks.
All policy replays had 24 periods, all constraint checks passed, and all
comparisons used common realized returns and common cost accounting.

This is post-holdout explanatory evidence. The locked holdout had already been
consumed before this dose was selected, so neither nominal p-values nor
confidence intervals restore a confirmatory superiority claim. The ten seeds
measure optimization variability, not ten independent market samples.

## Main 22-period economic results

| Strategy | CAGR | Volatility | Sharpe | Max DD | Annual CRRA CE | Monthly turnover |
|---|---:|---:|---:|---:|---:|---:|
| 100-path full state | 28.22% | 12.70% | 2.038 | 7.08% | 27.26% | 0.110 |
| 100-path no visible dependence | 25.15% | 12.02% | 1.940 | 5.91% | 24.31% | 0.109 |
| Matched 1,000-path full state | 25.75% | 12.30% | 1.938 | 5.94% | 24.87% | 0.347 |
| Matched 1,000-path no visible dependence | 23.23% | 12.27% | 1.775 | 6.09% | 22.37% | 0.375 |
| Historical only | 46.07% | 16.57% | 2.398 | 5.26% | 44.38% | 0.614 |
| Equal weight | 24.30% | 13.40% | 1.700 | 5.22% | 23.31% | 0.000 |
| DCC-GARCH | 28.60% | 13.54% | 1.941 | 5.79% | 27.53% | 0.153 |
| Static vine | 31.10% | 16.86% | 1.702 | 6.51% | 29.46% | 1.051 |
| Dynamic NN-vine optimizer | 29.76% | 16.43% | 1.679 | 6.78% | 28.21% | 1.087 |

The 100-path full-state ensemble is the stronger of the two new policies. It
has the lowest volatility and highest Sharpe among the full-state policy and
six financial benchmarks, while its return and CRRA CE remain close to DCC,
static vine, and the dynamic NN-vine optimizer. It is economically competitive
and implementation-efficient, but it does not dominate the benchmark set.

The no-visible-dependence ensemble is more defensive: it has 12.02% volatility
and 5.91% maximum drawdown, but gives up roughly 3.1 CAGR points and 3.0 CRRA CE
points to the 100-path full-state ensemble. It descriptively exceeds equal
weight, shrinkage mean-variance, and rolling vine in CRRA CE, while trailing
DCC, static vine, and dynamic NN-vine. None of its six benchmark advantages or
deficits was statistically established after Holm correction.

## Dose effects

Relative to the matched causal 1,000-path cohort, reducing both unique paths
and presentations from 1,000 to 100 increased annual CRRA CE by 2.39 percentage
points for the full-state ensemble and 1.94 points for the no-visible ensemble.
The 24-period sensitivity estimates are similar at +1.98 and +2.49 points.
Every confidence interval crosses zero, and adjusted one-sided p-values are
0.708 or larger. These are modest favorable point estimates, not established
dose effects.

The matched-seed evidence is directionally consistent but heterogeneous. Six
of ten full-state seeds and six of ten no-visible seeds improved on their
1,000-path counterparts. Mean seed-level CRRA changes were +2.87 and +2.28
points, respectively, but individual changes ranged from approximately -22 to
+21 points for full state and -42 to +51 points for no-visible dependence.
Thus, lower dose does not uniformly rescue optimization; seed sensitivity is
material.

The strongest dose effect is operational rather than predictive. Mean monthly
turnover across matched individual seeds fell from 0.545 to 0.216 for full
state and from 0.690 to 0.228 for no-visible dependence. All ten paired seeds
showed lower turnover. Mean gross exposure remained almost unchanged near
1.40, so the dose reduction regularized portfolio movement rather than the
chosen gross risk budget.

Against the original frozen 20-seed full ensemble, the 100-path full ensemble
is almost economically equivalent: CAGR is 28.22% versus 29.08%, Sharpe is
2.038 versus 2.018, and CRRA CE is 27.26% versus 28.05%. Its turnover is much
lower, 0.110 versus 0.317. This comparison is descriptive because the seed
cohort and ensemble size differ, but it shows that the original economic result
can be reproduced approximately with one tenth of the simulator exposure and
substantially less trading.

## Ensemble mechanism and allocation behaviour

The individual 100-path policies still operate near the intended risk limit:
mean gross exposure is 1.402--1.404 and mean short notional is approximately
0.201--0.202. They disagree strongly on direction. Average pairwise weight
correlation is only 0.263 for full state and 0.048 for no-visible dependence,
with some negative correlations.

Weight averaging cancels 99.6% of incremental gross/short exposure in the
full-state ensemble and 95.9% in the no-visible ensemble. Ensemble gross
exposure is only 1.001 and 1.016, respectively. Ensembling also reduces
turnover by approximately 49% and 52% relative to the corresponding mean seed
policy. The deployable ensemble is therefore a low-leverage consensus
portfolio assembled from individually leveraged and directionally diverse
policies. It must not be described as representative of a typical seed.

The full-state consensus is strategically interpretable: average allocation is
approximately 38.9% gold, 19.1% NASDAQ, 16.9% S&P 500, 11.8% Dow, and smaller
allocations to Chinese and dividend assets. The no-visible ensemble holds less
gold (27.2%) and more dividend and ChiNext exposure. The 100-path full-state
advantage is therefore associated with a more defensive gold-heavy consensus,
not with additional ensemble leverage.

## Historical-only comparison

Historical-only remains the strongest observed explanatory ensemble. Its
annual CRRA CE exceeds the new full-state and no-visible policies by 17.12 and
20.06 percentage points on 22 periods. The block-bootstrap intervals lie
entirely in the adverse direction for both synthetic candidates; the same
conclusion holds over all 24 locked periods. Historical-only also beats the
100-path full policy in eight of ten matched seeds by CRRA CE.

This gap cannot be dismissed as an uncharged-turnover artifact. Historical-only
turnover is much higher (0.614), and its implementation drag is approximately
2.99 return points versus 0.39--0.43 for the new ensembles, but common
accounting already charges those costs. It still wins on realized return,
Sharpe, drawdown, tail loss, and CRRA CE in this short sample.

Nor does this prove that historical-only is the true population optimum. The
comparison is post-selection on a consumed 22-period market path, with one
historical regime and overlapping training trajectories. Its advantage is
concentrated partly in several strong risk-on months, especially September
2024, August 2025, and April 2026. Nevertheless, the result is too broad and
cost-adjusted to label as mere luck without independent evidence. The correct
diagnosis is unresolved synthetic-to-real negative transfer.

## Representation interaction

At 100 paths, full state beats no-visible dependence by 2.95 annual CRRA CE
points, and seven of ten paired seeds favor full state. The confidence interval
crosses zero. This reverses the two-window retrospective ranking, where
no-visible dependence was strongest, and differs from the consumed-holdout
causal study, where scenario-CVaR-only was descriptively strongest.

The evidence therefore does not identify one universally optimal observation
representation. It does identify a robust research issue: high-dimensional
raw dependence information is not reliably valuable, scalar scenario risk can
be useful in some regimes, and the preferred representation interacts with
simulator exposure, optimization seed, and market window. This is a stronger
and more honest scientific result than selecting whichever representation wins
on one holdout.

## What the experiment does and does not identify

The v1 intervention jointly changed three quantities:

1. unique synthetic episodes: 1,000 to 100;
2. synthetic episode presentations: 1,000 to 100;
3. pretraining gradient updates: approximately 23,873 to 2,273 under the same
   replay warm-up rule.

Consequently, v1 cannot distinguish excessive simulator diversity from
excessive optimization on the simulator distribution. Calling the result
"overfitting to too much synthetic data" is premature. The data support a
broader negative-transfer explanation: longer simulator optimization creates
more active policies without delivering better historical transfer.

## Recommended final HPC experiment

Run one final two-by-ten matched-seed experiment using exactly the same 100
unique episodes but 1,000 episode presentations (ten deterministic, seeded
passes), for both full state and no-visible dependence. Keep the original
1,000-presentation exploration schedule, all architecture and fine-tuning
settings, seeds, costs, and evaluation accounting fixed.

Together, the three observed designs identify the mechanism:

- 1,000 unique / 1,000 presentations: original matched comparator;
- 100 unique / 100 presentations: completed v1;
- 100 unique / 1,000 presentations: proposed v2.

Comparing v2 with v1 isolates presentation/update budget at fixed diversity.
Comparing v2 with the original 1,000/1,000 design isolates synthetic diversity
at fixed presentation budget. Repeated episodes must be explicitly recorded as
100 unique episodes and 1,000 presentations; they must not be mislabeled as
1,000 independent paths.

If v2 approaches historical-only while preserving lower turnover, excessive
simulator diversity was the likely problem. If v2 falls back toward the
1,000/1,000 result, excessive simulator optimization is the more likely cause.
If v2 performs worse than both, repeated-path memorization is implicated. No
additional architecture or algorithm sweep is warranted before this mechanism
is resolved.

## Overarching paper storyline

The project does not validate universal vine-RL superiority. It validates the
need for scientifically controlled integration of dependence models and deep
portfolio policies.

First, the frozen main model demonstrates that a dynamic-vine recurrent RL
ensemble can be highly competitive and risk efficient under common realized
returns, costs, shorting, and leverage constraints. Second, the causal and
walk-forward studies show that more dependence information is not automatically
better: raw vine-state expansion can overload the controller, and the useful
representation is regime- and dose-dependent. Third, the dose experiment shows
that one tenth of the synthetic exposure can approximately preserve economic
performance while greatly reducing turnover and computation. Fourth, the
historical-only result exposes unresolved simulator-to-market transfer rather
than allowing synthetic fidelity diagnostics to stand in for decision value.

The contribution is therefore a validated dynamic-vine/RL research framework
and an empirical mechanism result: dependence simulation is potentially useful
as risk information and regularization, but its representation and training
dose must be controlled. Synthetic realism, statistical fidelity, and larger
training volume do not by themselves guarantee better portfolio decisions.
That conclusion is scientifically valuable because it replaces a fragile
"complexity wins" claim with reproducible evidence about when complexity helps,
when it harms, and how to diagnose the difference.
