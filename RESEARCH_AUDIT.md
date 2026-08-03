# Research audit and falsification protocol

Status: **code-corrected and smoke-tested; economic superiority not yet established**.

A model is not state of the art because its backtest is best. It earns that
claim only after the data-generating process, training protocol, portfolio
constraints, and inference survive the pre-declared checks below.

## Locked design

- Raw daily prices are validated as positive, finite, uniquely dated, and
  strictly ordered. The source file hash is stored in each run manifest.
- Exactly the final 24 holding periods are locked for one-time OOS evaluation.
  The last price observation is a holding-period end, never a no-op decision.
- Marginals, empirical transforms, D-vine order, static backbone, and every
  neural edge are fitted only through the training cutoff.
- All 1,000 generated paths are consumed once in synthetic pre-training.
- Historical fine-tuning uses all 61 currently available causal 30-month
  context plus 24-month decision trajectories in balanced sequential passes.
- The locked 24 months never enter pre-training, fine-tuning, normalization,
  marginal fitting, vine selection, or hyperparameter selection.

## Material defects found and disposition

| Severity | Defect | Consequence | Disposition |
|---|---|---|---|
| Critical | Final endpoint was also a decision date | One of 24 OOS months had no returns | Calendar-safe decision/end pairs; invariant tested |
| Critical | 30-state LSTM history was repeated padding while episodes had 24 actions | The network never saw a fully genuine sequence | Every episode now carries 30 strictly prior burn-in returns/states |
| Critical | Ablation and sensitivity results used assumed factors and random noise | Fabricated paper evidence | Removed; analyses accept completed run logs only and fail closed |
| Critical | Benchmark/RL comparisons used different realized data and leaky artifacts | Rankings were not interpretable | Legacy engine disabled; common weight-only scoring contract added |
| Critical | Dynamic vine changed only tree 1 | Pseudo-dynamic dependence | All 21 unconditional and conditional t edges now have dynamic NNs |
| Critical | Tree-1 truncation loses held-out likelihood (mean 0.1596, z 5.04) | Truncation is empirically indefensible here | Full six-tree D-vine retained; every retained edge dynamic |
| High | NN snapshot reused the last training feature, one day stale | Dependence forecast lagged the information set | Explicit next-step feature construction |
| High | Marginal failure only emitted a warning | Misspecified residuals contaminated the copula | 12-model primary grid plus causal component-EWMA fallback and hard gate |
| High | DIVIDEND squared residual dependence persisted | Incorrect uniforms/tails | Component EWMA yields Var(z)=1.0004 and minimum LB p=0.0123 |
| High | Daily conditional mean coefficients were applied to monthly returns | Mixed-frequency state noise | Daily conditional-mean feature removed |
| High | Synthetic paths were conditionally IID in time | LSTM could not learn historical serial structure | Monthly stationary AR copula added; temporal gate added |
| High | Gross-return means near one made fidelity trivial | Economically large drift errors could pass | Diagnostics use log returns and standardized errors |
| High | 1% monthly tails had about one observation | Unstable VaR/CVaR claims | Marginal tails use 5%; co-exceedance reports event counts and exact intervals |
| High | DDPG used one critic | Positive critic bias and instability | Replaced with recurrent TD3: twin critics, delayed actor, target smoothing |
| High | Shorting and leverage were free | Upward-biased leveraged strategies | Common turnover, stock-borrow and cash-financing costs |
| High | CAGR divided by volatility was called Sharpe | Incorrect performance statistic | Arithmetic excess-return Sharpe; CAGR reported separately |
| High | HAC test divided an intercept SE by sqrt(n) twice and was called DM | Invalid significance | Correct NW mean test, Holm control, moving-block Reality Check |
| Medium | Arbitrary D-vine order | Avoidable dependence loss | Exact 7! training-only order search by adjacent absolute Kendall tau |
| Medium | 2,000 CVaR scenarios were stored for unused burn-in steps | Multi-GB bundle waste | Burn-in stores one vector; default scenarios reduced to 512 |
| Medium | Model re-fit in generator, trainer and evaluator | Wasted compute and drift | Versioned all-tree NN fit persisted once and reloaded |
| Repository | Corrupted Git pack object | History/status/manifests cannot be trusted | Do not reset; recover from a known-good remote or clone separately |

## Statistical gates before RL

rl/synthetic_returns.r writes diagnostics and exits non-zero unless all hold:

1. Seven marginal log-return distributions pass standardized
   mean, volatility, and 5%-tail tolerances.
2. Every synthetic pairwise correlation lies in the historical moving-block
   bootstrap interval.
3. Every 5% lower-tail conditional co-exceedance interval overlaps its
   historical exact interval, with event counts reported.
4. Every asset's return and squared-return lag-1 discrepancy is within
   2 / sqrt(n).
5. The persisted vine has all d(d-1)/2 = 21 dynamic edges and the locked
   split metadata matches runtime.

These are necessary checks, not proof that generated data equal the unknown
population.

## Superiority experiment

1. Freeze code and configuration hashes.
2. Train at least 20 independent seeds for RL algorithm variability.
3. Generate weights for all benchmarks under the same information set.
4. Feed weights only to eval/research_protocol.r; it applies identical realized
   returns, exposure limits, turnover, borrow fees and financing.
5. Pre-declare the primary benchmark, CRRA gamma, primary utility statistic,
   and the single locked OOS analysis.
6. Report CAGR, arithmetic Sharpe, volatility, maximum drawdown, 5% CVaR,
   turnover, short notional, and compute time with confidence intervals.
7. Use paired HAC utility differences with Holm correction and the
   moving-block White Reality Check across tried models.
8. Treat rolling-origin folds before the locked endpoint as development
   evidence. Do not repeatedly inspect or tune on the final 24 months.
9. Report all seeds and configurations, not only the winning run.

A 24-month path has low power, especially for tail outcomes. Failure to reject
is not equivalence; a higher point estimate is not statistical superiority.

## Verification performed locally

- All 26 R files parse under R 4.6.1.
- Four embedded Python blocks compile.
- Fast research-protocol tests pass.
- Calendar: 2,945 daily returns, 114 training holding periods, 24 OOS periods.
- Real-data marginal gate passes 7/7.
- Static D-vine density equals the sum of recursively constructed pair-copula
  log densities to numerical precision.
- Fully dynamic all-tree NN-vine smoke fit/build passes.
- Nested NN model persistence round-trip passes.
- TD3 twin-critic update and checkpoint round-trip passes on PyTorch 2.13 CPU.
- Causal burn-in, long/short constraints, borrowing cost and CRRA telescoping
  invariants pass.
- End-to-end generator smoke run reached artifact serialization and failed only
  the intended fidelity gate at the deliberately small 10-path/one-epoch setup.
- Full synthetic generation and 1,000-episode RL training have not been run in
  this desktop audit.

## Commands

~~~powershell
& 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe' --vanilla tests\run_tests.r
python tests\check_embedded_python.py
& 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe' --vanilla rl\synthetic_returns.r config\config.yaml
& 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe' --vanilla run_with_config.r config\config.yaml
& 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe' --vanilla evaluate_with_config.r config\config.yaml
~~~

## Scientific basis

- Fujimoto, van Hoof and Meger (2018), Addressing Function Approximation Error
  in Actor-Critic Methods.
- Almeida, Czado and Manner (2016), Modeling high-dimensional time-varying
  dependence using D-vine SCAR models.
- Rockafellar and Uryasev (2000/2002), optimization of conditional value at
  risk.
- White (2000), A Reality Check for Data Snooping.
- Hansen (2005), A Test for Superior Predictive Ability.

The empirical truncation failure is why this implementation does not adopt the
otherwise defensible dynamic-first-tree, static-higher-tree shortcut.
