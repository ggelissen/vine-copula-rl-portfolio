# Locked OOS v4: independent publication decision

## Decision

**Outcome: mixed result.**

The predeclared 20-seed NN-vine LSTM-TD3 ensemble is economically competitive,
has the lowest observed volatility and the highest observed Sharpe ratio, and
obeys all portfolio constraints. It does **not** establish statistical
superiority, does **not** maximize the primary CRRA economic objective, and does
**not** yet establish the incremental value of RL over the non-RL NN-vine.

The successful v4 result must be preserved unchanged. Model or hyperparameter
tuning against these 24 locked periods would consume them as development data.
Any redesigned model needs new confirmatory evidence from locked walk-forward
windows, another market, or future observations.

## Source integrity and independent reconciliation

- Supplied archive SHA-256:
  `770d2944f915d0ad21ae9af32e31d68d652fdb54e98939caeab45c327b4e5ea1`.
- The computed archive hash exactly matches the supplied sidecar.
- Locked batch status: complete; 20 full policies; six benchmarks; one
  predeclared mean-weight ensemble.
- The raw panel contains 27 strategies x 24 periods = 648 rows.
- Every strategy uses the same dates, asset order, and realized seven-asset
  return panel.
- Independent gross-return, cost, net-return, wealth, and headline-metric
  recomputation agrees to floating-point precision.
- All 20 checkpoints and all 20 seed weight paths have distinct hashes.
- The ensemble weights equal the arithmetic mean of the 20 seed weights to
  machine precision.
- All 648 strategy-period rows satisfy net, gross, long, and short constraints
  at the declared tolerance.

The primary sample contains **22 complete periods**, not 24. The excluded rows
are 30 January to 27 February 2026 (14 joint trading observations) and 30 June
to 6 July 2026 (three observations). The second is genuinely partial. The
first spans a normal month but fails the predeclared 15-joint-observation rule;
that rule is conservative and should be reconsidered for future cross-market
protocols.

## Primary 22-period economic results

| Strategy | Total return | CAGR | Volatility | Sharpe | Sortino | Max DD | Omega | Annual CRRA CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Equal weight | 49.16% | 24.37% | 13.42% | 1.702 | 5.029 | 5.22% | 4.141 | 23.38% |
| Shrinkage mean-variance | 48.69% | 24.16% | 14.17% | 1.608 | 2.952 | 7.83% | 3.363 | 22.98% |
| DCC-GARCH | 58.79% | 28.69% | 13.55% | 1.944 | 4.632 | 5.79% | 4.965 | 27.62% |
| Static vine | **64.50%** | **31.19%** | 16.88% | 1.704 | 4.478 | 6.51% | 3.756 | **29.55%** |
| Rolling vine | 40.81% | 20.52% | 18.94% | 1.082 | 2.033 | 9.90% | 2.359 | 18.53% |
| Dynamic NN-vine, no RL | 61.44% | 29.86% | 16.45% | 1.682 | 4.300 | 6.78% | 3.760 | 28.30% |
| NN-vine LSTM-TD3 ensemble | 59.67% | 29.08% | **13.19%** | **2.018** | 4.642 | 6.64% | 4.880 | 28.05% |

The ensemble ranks first in Sharpe and volatility, second in Sortino and Omega,
third in total return and CRRA CE, fourth in drawdown/CVaR, and fifth in Calmar.
Its strongest defensible contribution is risk and implementation efficiency,
not maximum wealth or utility.

On an initial wealth of 100,000, ensemble terminal wealth is 159,668.87:

- +10,504.49 versus equal weight (+7.04% relative terminal wealth);
- +10,981.78 versus shrinkage mean-variance (+7.39%);
- +876.88 versus DCC-GARCH (+0.55%);
- -4,832.60 versus static vine (-2.94%);
- +18,857.95 versus rolling vine (+13.39%);
- -1,773.23 versus dynamic NN-vine without RL (-1.10%).

The 24-row shortened-period sensitivity makes the ensemble rank first, but the
ranking reversal is driven materially by the final three-day row, in which the
static and dynamic NN-vine strategies lose about 3% while the ensemble is
approximately flat. It is descriptive sensitivity evidence, not a replacement
for the predeclared primary sample.

## Statistical inference

The preregistered primary comparison fails:

- Mean monthly CRRA utility difference versus equal weight: `0.0030341`.
- 95% paired circular-block-bootstrap interval: `[-0.0056916, 0.0101156]`.
- One-sided bootstrap p-value: `0.2347`.
- Primary-family Holm-adjusted bootstrap p-value: `0.9388`.

All six ensemble-versus-benchmark CRRA intervals contain zero:

| Benchmark | Utility effect | 95% block-bootstrap interval | One-sided p | Holm p |
|---|---:|---:|---:|---:|
| Equal weight | 0.003034 | [-0.005692, 0.010116] | 0.2347 | 0.9388 |
| Shrinkage mean-variance | 0.003303 | [-0.001176, 0.008345] | 0.0928 | 0.5568 |
| DCC-GARCH | 0.000273 | [-0.004230, 0.005428] | 0.4406 | 1.0000 |
| Static vine | -0.000951 | [-0.009173, 0.008199] | 0.5639 | 1.0000 |
| Rolling vine | 0.006323 | [-0.006399, 0.018376] | 0.1645 | 0.8225 |
| Dynamic NN-vine, no RL | -0.000158 | [-0.009793, 0.010222] | 0.5024 | 1.0000 |

White's reality check selects the static vine as the best observed candidate
against equal weight, but its p-value is `0.5142`. No screened strategy has a
statistically established advantage after data-snooping control.

With only 22 observations, a 5% empirical CVaR is based on one tail event and
equals the worst observed month. The ensemble's realized 5% VaR and CVaR are
5.29% and 6.64%; these quantities are descriptive and cannot support a tail-risk
superiority claim. Sharpe, skewness, kurtosis, drawdown, and tail estimates all
remain highly uncertain at this sample size.

## Seed robustness and the ensemble mechanism

The ensemble does not represent a typical seed policy.

| Metric | Worst seed | Median seed | Best seed | Ensemble |
|---|---:|---:|---:|---:|
| CAGR | 15.47% | 25.95% | 45.46% | 29.08% |
| Sharpe | 0.983 | 1.628 | 2.237 | 2.018 |
| Max drawdown | 12.15% | 8.10% | 5.45% | 6.64% |
| Terminal wealth | 130,169.62 | 152,677.09 | 198,765.78 | 159,668.87 |
| Monthly turnover | 1.001 | 0.539 | 0.338 | 0.317 |

Only 11/20 seeds beat equal weight on CAGR, 9/20 on Sharpe, and no seed beats
equal weight on drawdown or empirical CVaR. Only 8/20 beat the static vine on
CAGR and 9/20 on Sharpe. Only 9/20 beat the dynamic NN-vine on CAGR or Sharpe.
This is not seed-robust superiority.

Cross-seed averaging materially changes the portfolio:

| Quantity | Mean individual seed | Ensemble |
|---|---:|---:|
| Gross exposure | 1.4023 | 1.0221 |
| Short notional | 20.12% | 1.11% |
| Monthly turnover | 0.5747 | 0.3170 |
| Transaction cost | 5.747 bps/month | 3.170 bps/month |
| Financing cost | 5.029 bps/month | 0.277 bps/month |

About 94.5% of incremental gross/short exposure cancels because seed policies
take opposing positions. The ensemble's best-in-sample Sharpe is therefore
substantially an ensemble diversification, de-leveraging, and cost-compression
result. This is valid because the mean-weight ensemble was predeclared, but the
paper must distinguish the deployable ensemble from a representative learned
long-short policy.

The ensemble's mean allocation is approximately 33.1% Gold, 26.9% NASDAQ,
14.1% DOW, 13.1% S&P 500, 7.8% ChiNext, 2.9% Dividend, and 2.3% SSE50. It is
mostly long and close to unlevered out of sample. Gross-return contributions are
concentrated in Gold, NASDAQ, and ChiNext during a strong realized market
period; this does not establish crisis robustness.

## Implementation and constraints

The ensemble has mean/max gross exposure 1.022/1.121, mean short notional
1.11%, and turnover 0.317 per month (3.804 annualized). Its average transaction
and financing charges are 3.170 and 0.277 basis points per month. Net total
return is 1.215 percentage points below its 60.884% pre-cost total return, about
2.0% of the pre-cost gain.

The static, rolling, and dynamic NN-vine optimizers have roughly 1.44-1.48 mean
gross exposure, 22-24% mean short notional, and 1.05-1.24 monthly turnover.
Their implementation drags are roughly 5.8-6.0 total-return points. The ensemble
therefore narrows its gross-performance deficit through lower costs and risk.

The frozen scorer measures transaction turnover from one target weight vector
to the next and does not first drift the old weights through realized returns.
A deterministic post-hoc drift-aware check increases ensemble turnover from
0.3170 to 0.3303 and lowers total return by only 0.047 percentage points; all
rankings and conclusions are unchanged. Future protocols should nevertheless
use drift-aware pre-trade holdings.

All declared position constraints pass. Across all 648 rows, maximum net error
is `9.92e-9`, maximum gross is `1.5000000098`, maximum long weight is 0.60, and
minimum short weight is -0.20; there are no violations at tolerance `1e-6`.

## Benchmark convergence

There is no silent fallback and every benchmark portfolio is finite and
feasible. Input dates never exceed their decision dates. However, 12 of 120
optimized benchmark decisions stopped with NLopt code 5 at the declared 2,000
evaluations: four static-vine, four rolling-vine, and four dynamic-NN-vine
solves. Code 5 means maximum evaluations were reached, not that the XTOL
criterion was reached. Feasibility is proven; numerical optimality is not.

The current result must not be silently replaced by a post-holdout rerun with a
larger evaluation budget. Future protocols should fail closed on max-evaluation
stops, log solver messages/KKT residuals, and use preregistered deterministic
multi-start or convergence robustness before accessing new confirmatory data.

## Provenance and retry disclosure

The successful archive is internally coherent but is not a self-contained
reconstruction package. It does not contain the frozen evaluation source
snapshot, complete evaluation release, training release, checkpoints, raw price
file, NN-vine fit directory, R package inventory, or container lock. The
`config_sha256` and `code_sha256` fields in the strategy manifest both contain
the same aggregate evaluation-release identifier rather than semantically
separate hashes.

The paper must also disclose that operational v2/v3 attempts accessed the same
holdout before v4 completed. The available failed archives expose the same
realized panel and some/all frozen weights. Their overlapping artifacts are
byte-identical to v4, so there is no evidence of intervening tuning, but strict
"first untouched access" language is no longer defensible. Describe the
sequence as frozen operational retries and publish an incident timeline.

Before submission, assemble a supplementary provenance package containing:

1. v4 archive and sidecar;
2. frozen evaluation release and sidecar, including source snapshot;
3. frozen training release and sidecar, checkpoints, gates, and manifests;
4. raw-market-data hash/snapshot or immutable licensed-data reference;
5. training marginal and NN-vine fit hashes;
6. evaluation and benchmark contracts;
7. all failed retry archives and an incident timeline;
8. R/Python/package/container versions and deterministic command record;
9. an internal contents hash manifest for the package.

## Figure interpretation

- Wealth: the ensemble is competitive, but static vine finishes highest in the
  primary scope. Curves co-move strongly on one market path.
- Drawdown: equal weight and DCC have shallower worst drawdowns than the
  ensemble; rolling vine is materially worse.
- Risk-return: the ensemble occupies the low-volatility/high-Sharpe corner;
  static vine occupies the high-return/high-volatility corner.
- Allocation heatmap: this is the arithmetic ensemble, not a typical seed.
  Add cross-seed allocation dispersion to avoid masking disagreement.
- Implementation: ensemble leverage and shorts are intermittent and small,
  while turnover remains economically relevant.
- Seed robustness: retain this in the main paper; dispersion is a principal
  result, not a footnote.
- Utility effects: every interval crosses zero and the figure directly rejects
  a superiority narrative.
- Monthly excess-return heatmap: seven cells are color-saturated by the
  percentile clipping rule; disclose clipping or annotate the extremes.

## Defensible paper claim

> In a frozen 22-complete-period out-of-sample comparison, the predeclared
> 20-seed mean-weight NN-vine LSTM-TD3 ensemble achieved the highest observed
> Sharpe ratio and lowest volatility while remaining implementation-efficient.
> It ranked third in terminal wealth and CRRA certainty-equivalent return.
> Its CRRA advantage over equal weight was positive but not statistically
> significant, and neither broad benchmark superiority nor incremental RL
> value over the static and dynamic non-RL vine optimizers was established.

Do not claim statistically proven superiority, robust tail-risk dominance,
typical-seed dominance, crisis resilience, or proven incremental NN-vine/RL
novelty from this result.

## Protocol-safe next actions

1. **Freeze and disclose this mixed result.** Do not retrain against it.
2. **Repair provenance without rescoring.** Build the supplementary package and
   operational-retry timeline described above.
3. **Run explanatory ablations.** Complete the preregistered no-vine TD3
   ablation; compare ensemble versus individual/median-seed deployment; then
   isolate vine state, LSTM memory, CVaR state/reward, transaction-cost reward,
   leverage/shorting, and fine-tuning. These are explanatory secondary results,
   not a second fresh confirmation on the same holdout.
4. **Use the already evaluated dynamic NN-vine without RL as the direct RL
   increment benchmark.** The current result is a practical low-risk tie, not
   evidence that RL adds return or CRRA utility.
5. **Preregister sensitivity analyses on training-prefix validation only.** Fix
   grids and decision rules before looking at any new test window.
6. **Create new confirmatory evidence.** Use multiple locked expanding/rolling
   windows, external markets, or future observations. Report window-level
   paired effects and meta-analytic aggregation rather than treating seeds as
   market replications.
7. **Harden the next evaluator.** Require explicit fixed dates, drift-aware
   turnover, day-prorated partial-period financing, complete artifact hashes,
   fail-closed solver convergence, KKT diagnostics, and environment lockfiles.

