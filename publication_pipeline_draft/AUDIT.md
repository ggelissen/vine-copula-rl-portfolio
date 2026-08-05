# Evaluation audit before locked holdout use

## What is already sound

- `evaluate_with_config.r` refuses holdout access unless training diagnostics,
  fine-tuning selection, and the no-holdout numerical/behavioural gates exist
  and pass.
- `rl/evaluate_rl.r` uses the final 24 realized holding periods, frozen
  training-only marginal transforms and NN-vine parameters, causal recurrent
  burn-in, CPU evaluation, deterministic actors, and schema-checked
  checkpoints.
- At a decision date, the dynamic vine state uses innovations ending on that
  date; the subsequently realized holding-period return is not included in the
  state used to choose its own weight.
- The RL environment and `eval/research_protocol.r` use the same return and
  exponential cost equations and both start turnover from equal weight.
- The legacy benchmark engine is disabled by default because it mixes horizons,
  simulations, constraints, and costs. The current ablation and sensitivity
  readers also fail when completed artifacts are missing rather than inventing
  metrics.

## Publication blockers found

1. **No valid benchmark weight engine is connected.** The live evaluator emits
   pretrained/full RL weights only. `eval/research_protocol.r` can score named
   weights, but nothing supplies the predeclared equal-weight, sample
   mean-variance, DCC-GARCH, static-vine, rolling-vine, and non-RL dynamic-vine
   logs under the common mandate.
2. **The common R validator omits single-asset caps.** It checks net exposure and
   gross leverage but not the configured +0.60/-0.20 limits. A benchmark can
   currently obtain an infeasible advantage. The draft evaluator enforces all
   four constraints.
3. **The 20 seeds have no locked batch evaluator.** `evaluate_with_config.r`
   overwrites `EVAL_MODEL_DIR` from one YAML file. Running it interactively seed
   by seed invites partial inspection. After the sweep, create all per-seed
   evaluation configs first, execute them as one frozen batch, and inspect only
   after every technically valid run finishes.
4. **The last alleged monthly period is shortened.** The manuscript records a
   30 June 2026 decision ending 6 July 2026. Treating that as a full monthly
   observation biases annualised summaries. The draft exports both
   `complete_periods` (primary) and `locked_all` (disclosed robustness) scopes.
5. **A moving `tail(24)` split is not a permanent lock.** Appending prices changes
   the train/evaluation boundary. The final integration must store explicit
   cutoff and holding-period dates in the run manifest and reject any different
   panel unless a new experiment version is declared and retrained.
6. **Risk-free rate and primary comparisons are implicit in live code.** The R
   metric function defaults to zero and the reality-check benchmark is the
   first list element. Both are explicit, hashed fields in the draft contract.
7. **Tail risk has very few observations.** A 5% empirical CVaR from 23--24
   months is determined by roughly one or two months. The draft exports the
   tail-event count and labels realized CVaR as descriptive; ex-ante vine CVaR
   remains a distinct state/reward diagnostic.
8. **Repeated training seeds are not market replications.** Averaging seed
   metrics and using their standard error as a performance significance test
   would be invalid. The draft reports a seed distribution and uses a
   predeclared, investable mean-weight ensemble for paired market-path tests.
9. **Evaluation provenance is incomplete.** Current RL CSVs do not carry input
   hashes, contract hashes, complete-period flags, or immutable output
   semantics. The draft records hashes and refuses an existing output folder.
10. **Current plots are diagnostic, not paper-ready.** They use step numbers,
    show only RL checkpoints, and omit drawdown, implementation costs,
    uncertainty, and complete-period disclosure. The draft figure layer fixes
    these omissions.
11. **The manuscript's current-status paragraph is stale.** It says the current
    ablation/sensitivity scripts manufacture metrics, but those readers now
    fail closed on missing completed logs. Keep the results placeholders until
    experiments exist, then update that description to match the corrected
    code rather than repeating the obsolete warning.

## Work deliberately deferred until the sweep ends

- Do not edit `rl/evaluate_rl.r`, `evaluate_with_config.r`,
  `eval/research_protocol.r`, `config/config.yaml`, or benchmark files while the
  sequential runner may source them for later seeds.
- After the sweep, merge the stricter validator and immutable batch runner into
  the live tree, then implement and test causal benchmark weight generators.
- Do not use old `benchmark_results.RData`, `evaluation_results.RData`, or
  existing wealth figures in the paper; they predate the common protocol.
