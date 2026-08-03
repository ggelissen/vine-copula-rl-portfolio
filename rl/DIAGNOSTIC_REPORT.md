# RL diagnostic and corrected protocol

> Historical audit note only. The current authoritative design and validation
> status are in ../RESEARCH_AUDIT.md; model names and commands below may refer
> to the superseded DDPG/tree-1 implementation.

## What made the previous result unreliable

1. `RLEnvironment$reset()` reset the precomputed-return cursor to one.  Thus
   each nominally independent episode reused the first 24 matrices of a much
   larger bundle.  The `full-a100-32cpu-seed-20260741` training log consequently
   shows rewards converging to nearly identical values instead of learning over
   diverse market paths.
2. `DDPGAgent$select_action()` ignored `noise_scale`, so there was no action
   exploration.  `hidden`, `num_layers`, replay capacity, entropy coefficient,
   and gradient-clipping configuration were also partly ignored.
3. The environment produced a terminal-only reward; `lambda` and `kappa` had
   no effect.  It computed vine features but omitted them from the observation,
   and omitted the previous portfolio although turnover depends on it.
4. The previous evaluation compared RL wealth on freshly simulated paths with
   benchmark wealth on one realised historical path.  For seed 20260741, the
   reported full-RL final wealth was 98,866 versus 164,702 for Rolling Vine MV,
   but that is not an apples-to-apples performance statistic.
5. The simulator's marginals are daily but episode steps and realised benchmark
   wealth are monthly.  The old square-root aggregation generated badly
   distorted monthly tails and volatility; synthetic vine draws are now mapped
   to the empirical in-sample monthly marginal distributions instead.
6. The full-sample marginal artifact could leak the holdout through GARCH
   parameters and empirical residual ranks.  The data-preparation step now
   writes `data/training_marginal_results.RData`: AR-GARCH models are fitted on
   the training prefix only, and later pseudo-observations are filtered with
   those fixed parameters and the training residual CDF.
7. The marginal-fitting helper referred to an undefined `returns` object when
   determining asset names; it now uses its locally loaded `raw_returns`.

## Corrected experiment

1. Regenerate the bundle.  The final 24 monthly periods are held out; each
   synthetic pre-training episode has a matching dynamic-vine start, while
   fine-tuning episodes use realised historical holding returns only.

   `Rscript --vanilla rl/synthetic_returns.r`

2. Train from the configuration.

   `Rscript --vanilla run_with_config.r config/config.yaml`

3. Evaluate the newly trained checkpoint on the same realised 24-month path
   as the benchmarks.

   `Rscript --vanilla evaluate_with_config.r config/config.yaml`

Old checkpoints intentionally fail the new evaluator: their observation
dimension and checkpoint metadata predate the corrected state/reward design.

## Before making a publication claim

Use a rolling or expanding walk-forward design with several non-overlapping
test windows.  Choose hyperparameters only on validation windows and report
each seed/window result, turnover, costs, drawdown, and a paired performance
test against every benchmark.  A single 24-month holdout is useful for a smoke
test but is not enough evidence that the RL model is the best strategy.
