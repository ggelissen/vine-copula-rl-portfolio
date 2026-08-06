# Dynamic Portfolio Selection with Vine Copulas and Recurrent TD3

Research implementation of a constrained dynamic portfolio-allocation pipeline combining AR-GARCH marginals, a neural time-varying D-vine, synthetic pretraining, historical fine-tuning, and recurrent Twin Delayed DDPG (TD3).

> **Research status:** synthetic-data validation and the preregistered 20-seed training sweep are complete. Ablation and benchmark evaluation are in progress; the locked final out-of-sample evaluation has not yet been opened. Economic superiority is not established.

## Research question

Can a recurrent portfolio policy use time-varying cross-asset dependence information to improve risk-adjusted allocation under realistic leverage, shorting, turnover, and tail-risk constraints?

The project is designed around a stricter companion question: can any apparent improvement survive leakage controls, common realised-return scoring, multiple RL seeds, benchmark comparison, and predeclared statistical checks?

## Method at a glance

```text
Daily asset prices
    ↓
Training-only AR-GARCH marginals and pseudo-observations
    ↓
Fully dynamic seven-asset D-vine dependence model
    ↓
Validated synthetic monthly episodes
    ↓
Recurrent TD3 pretraining
    ↓
Causal historical fine-tuning
    ↓
Locked 24-period out-of-sample evaluation
    ↓
Common realised-return scoring against benchmarks
```

### Dependence and marginal modelling

- AR(1) marginal models with sGARCH, GJR-GARCH, and eGARCH candidates
- Skewed Student-t innovations and residual diagnostics
- Training-only empirical transforms to avoid holdout leakage
- Exact training-only D-vine order search for seven assets
- Neural time variation across all 21 unconditional and conditional D-vine edges
- Fidelity gates for marginal moments, correlation, finite-sample tail co-exceedance, and temporal dependence

### Portfolio policy

- Recurrent TD3 with twin critics, delayed policy updates, and target smoothing
- Long/short portfolio projection with explicit net and gross exposure constraints
- Position limits, transaction costs, stock-borrow costs, and financing costs
- Terminal-wealth CRRA objective expressed through dense telescoping increments
- Scenario-based 95% CVaR penalty
- Behavioural gates for leverage, diversification, turnover, and constraint compliance

### Evaluation design

- Final 24 holding periods locked away from model fitting and tuning
- Synthetic-only pretraining
- Historical fine-tuning restricted to the training prefix
- One realised historical return path shared by RL and benchmark strategies
- Equal-cost and equal-constraint scoring across methods
- Planned multi-seed uncertainty, HAC utility comparisons, multiplicity control, and a moving-block Reality Check

## Evidence status

| Component | Status |
|---|---|
| Calendar and holdout invariants | Verified by fast tests |
| Training-only marginal and dependence protocol | Implemented with runtime checks |
| Dynamic all-tree D-vine smoke fit and persistence | Smoke-tested |
| Recurrent TD3 projection and checkpoint invariants | Smoke-tested |
| Full synthetic fidelity run | Pending |
| Preregistered multi-seed training | Pending |
| Locked out-of-sample comparison | Pending |
| Economic superiority claim | Not established |

See [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md) for the defect history, falsification protocol, and publication gates. Historical diagnostic notes under `rl/` are retained for traceability; the root audit is authoritative when the files disagree.

## Repository map

```text
benchmark_models/   benchmark portfolio and dependence models
config/             master configuration, experiment manifests, environment snapshots
data/               input datasets and generated artifacts
eval/               common scoring, statistics, ablation and sensitivity runners
helper/             data loading, time splitting, marginals and reproducibility helpers
hpc/                cluster launch scripts
paper_revision/     code-faithful manuscript and compiled PDF
rl/                 generator, environment, recurrent TD3 training and evaluation
tests/              R and Python invariants
```

## Reference workflow

The project uses both R and Python. Exact package snapshots for the publication configuration are stored under `config/freeze_schema5/`.

```bash
# Fast protocol and calendar checks
Rscript --vanilla tests/run_tests.r
python tests/check_embedded_python.py
python tests/test_leverage_gate.py

# Generate and validate synthetic training episodes
Rscript --vanilla rl/synthetic_returns.r config/config.yaml

# Train without opening the locked evaluation sample
Rscript --vanilla run_with_config.r config/config.yaml

# Run only after the model and protocol have been frozen
Rscript --vanilla evaluate_with_config.r config/config.yaml
```

The full workflow is computationally intensive and was designed for an HPC/GPU environment. Review `config/config.yaml` before running: device, core count, artifact paths, and publication seeds are explicit configuration values.

## Data protocol

The repository contains three asset-universe CSV files. Before redistributing or using them, verify the source and licensing terms and document them in `data/README.md`. Each experiment stores hashes and split metadata so a run can be tied to its exact inputs.

The final 24 monthly holding periods are evaluation-only. They must not enter marginal fitting, D-vine fitting, normalisation, synthetic pretraining, historical fine-tuning, or hyperparameter selection.

## Limitations

- A single 24-period market path has low statistical power, especially for tail outcomes.
- Synthetic fidelity checks are necessary but cannot prove that the generator matches the unknown data-generating process.
- Repeated RL seeds quantify optimisation uncertainty, not independent market histories.
- Copula features primarily describe dependence and risk; they do not create return predictability by themselves.
- The current repository includes historical/rejected artifacts for research traceability. Do not interpret a checkpoint or log as an endorsed result unless it passes the current schema and publication gates.

## Paper

The code-faithful manuscript and compiled PDF are under `paper_revision/`. Result tables intentionally remain incomplete until the locked evaluation programme has been executed.

## Licence and disclaimer

No project-wide licence is currently declared. Until a licence is added, copyright remains with the relevant authors and no permission to reuse the code is implied.

This repository is academic research software, not investment advice. It is not a live trading system and makes no claim of future profitability.
