# Synthetic-data audit and data protocol

> Historical audit note only. The current authoritative gates are implemented
> in synthetic_returns.r and summarised in ../RESEARCH_AUDIT.md. The old 1%
> thresholds and iid-path statements below are superseded.

## Verdict on the supplied diagnostics

The supplied synthetic data must **not** be used for training.  Its comparison
is frequency-inconsistent: `historical_train` has 2,497 daily observations,
whereas `pretrain` has 24,000 monthly episode steps.  Even after aggregating
the historical prices to the matching monthly holding period, the old
synthetic data is materially distorted.

| Example | Historical monthly | Old synthetic | Consequence |
|---|---:|---:|---|
| SP500 volatility | 4.56% | 6.67% | 46% too high |
| NASDAQ volatility | 5.43% | 8.17% | 50% too high |
| GOLD volatility | 3.43% | 6.39% | 86% too high |
| SP500 1% CVaR gross return | 0.882 | 0.769 | losses much too severe |
| NASDAQ 1% CVaR gross return | 0.874 | 0.742 | losses much too severe |

The old synthetic means are also systematically too high.  For example, its
SP500 gross mean is 1.0121 versus roughly 1.0095 in the comparable historical
monthly sample.  The `final_multiple` values (for example `1.20e+97`) are not
portfolio evidence: they result from compounding unrelated simulated episodes
as though they were one continuous century-long path.

The old data had numerical diversity, but it was not representative.  It was
therefore learnable in the undesirable sense that an agent could learn an
overly volatile, high-drift artificial environment.  It could not provide
credible evidence that a policy will transfer to the real market.

## Corrected protocol

1. Refit AR-GARCH marginals and construct pseudo-observations only from the
   in-sample prefix before the final 24 rebalancing months.  Later observations
   are filtered with those fixed parameters and the training residual CDF.
2. Pre-train only on newly simulated vine-copula returns.  The generated
   realised row is never an observed historical return.
3. Fine-tune only on overlapping 24-month episodes whose realised row is the
   historical monthly return.  Scenario rows are ex-ante copula simulations
   used only to estimate CVaR.
4. Evaluate only on the final 24 historical rebalancing periods.

The generator now maps simulated copula uniforms to the empirical **monthly**
in-sample marginal distributions.  This removes the erroneous daily-to-monthly
square-root scaling and makes drift, volatility, and marginal tails directly
testable. It retains an NN-driven dynamic-vine dependence regime in every
synthetic episode.

## Acceptance checks after regeneration

Do not train until `data/synthetic_diagnostics/fidelity_metrics.csv` has a
pass for every asset.  The script requires no more than 5% relative mean error
and 10% relative error in volatility, 1% VaR, and 1% CVaR.  Then inspect
`correlation_comparison.csv`: `pass_correlation` is a strict 0.10
absolute-error target and `statistically_compatible` accounts for finite
historical-sample uncertainty.  Persistent failures of both fields indicate
inadequate monthly dependence calibration.  `tail_dependence_comparison.csv`
reports the finite 5% conditional co-exceedance probability, exact historical
confidence intervals, and the number of historical tail events.  With about
115 monthly observations there are only around six 5% tail events: a
historical value of zero is not evidence of zero tail dependence and must not
be fitted as one. `temporal_dependence.csv` shows the remaining marginal
time-series gap. The NN vine changes cross-sectional dependence within each
synthetic episode, but the simulated marginal shocks remain conditionally IID;
persistence is learned primarily during historical fine-tuning and must be
reported honestly.

Finally, realistic data is not automatically a source of alpha.  A copula
mainly provides risk/dependence information.  Synthetic pre-training can teach
diversification and tail-risk control; genuine predictability must be learned
from the historical fine-tuning stage and demonstrated in the held-out test.

The marginal fitter now compares sGARCH, gjrGARCH, and eGARCH specifications,
selecting the lowest-BIC model that passes both residual and squared-residual
Ljung-Box checks when possible.  If it prints a warning that no candidate
passes, retain that asset as an unresolved data-model limitation rather than
moving directly to RL tuning.
