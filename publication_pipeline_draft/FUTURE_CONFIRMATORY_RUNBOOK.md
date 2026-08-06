# Future confirmatory evaluation runbook

## Purpose and status

The completed `locked_oos_v1` evaluation is consumed. Its realized outcomes may
now inform a new model-development cycle, but they cannot become a second test of
the same claim. This runbook governs genuinely new observations after the last
consumed holding end, 2026-07-06. It does not modify, reinterpret, or rerun the
frozen v4 evaluation.

`future_confirmatory_protocol.py` is a preregistration validator, one-access
orchestrator, and accounting auditor. It deliberately does not train models or
choose hyperparameters. Before execution, the research team must provide a
frozen test runner, frozen model releases, fixed train/validation/test data
snapshots, package locks, and a runtime attestation. The example contract uses
placeholder paths and digests and is not an executable research release.

The structural example contains only three periods per window so it remains
readable. A real publication contract should use enough non-overlapping periods
to meet an ex-ante power calculation. The 22-observation v4 result indicates
that another very short window will not establish broad superiority.

## Scientific rules

1. List every consumed holdout and its result hash. A future test interval may
   share a decision boundary with a consumed interval, but it may not reuse any
   realized holding return.
2. Fix the ordered asset universe, decision dates, holding-end dates, costs,
   leverage, position limits, primary strategy, primary benchmark, outcome,
   alpha, and multiplicity rule before test access.
3. Enforce `train_end < validation_start <= validation_end <= test_start` for
   every window. All model and hyperparameter selection ends no later than the
   validation end. Earlier test observations may update a preregistered state or
   causal filter, but may never trigger architecture, feature, hyperparameter,
   seed, stopping, or benchmark selection.
4. Use at least two non-overlapping future test windows. Pool only paired
   non-overlapping returns under the preregistered rule. Multiple training seeds
   estimate optimization variability; they are not independent market samples.
5. Execute target portfolios from drifted pretrade weights. For asset gross
   returns `g[t-1]` and prior target weights `w[t-1]`, the next pretrade weight is

   ```text
   pretrade_w[i,t] = w[i,t-1] * g[i,t-1] /
                     sum_j(w[j,t-1] * g[j,t-1])
   turnover[t] = sum_i(abs(target_w[i,t] - pretrade_w[i,t]))
   ```

   The first pretrade portfolio is the fixed equal-weight initial position.
6. Prorate short-borrow and cash-financing costs by actual calendar days:

   ```text
   financing_cost[t] = calendar_days[t] / day_count_basis *
       (annual_short_rate * short_notional[t] +
        annual_cash_rate * cash_borrow_notional[t])
   ```

   A three-day partial period therefore receives three days of financing, not
   one twelfth of the annual rate. Any performance annualization must also
   distinguish full monthly periods from partial periods.
7. Maintain separate SHA-256 inventories for code, configuration, data, models,
   and environment/package locks. Reusing one omnibus digest for all categories
   is not sufficient provenance.
8. Pin Python and R package closures, the exact runtime versions, platform, and
   container image digest. Locked test execution has no package installation or
   network access.
9. Accept only NLopt convergence codes 1 through 4. Code 5 means
   `MAXEVAL_REACHED` and is a failed solve even though it is positive. Missing
   audit rows, stale weights, fallback weights, clipping, or projection after a
   failed solve invalidate the entire locked batch.
10. A test-access ledger is created atomically before the orchestrator opens or
    hashes test data. Success and failure are archived. A failed accessed run is
    not permission to change code and try another output directory.

## 1. Preregister the design without opening test data

Copy the example to a new versioned path and replace every placeholder. Keep the
example unchanged.

```bash
cp publication_pipeline_draft/config/future_confirmatory_contract.example.json \
  frozen_releases/future_confirmatory_v1.json
```

The example's asset order and v4 economic mandate are illustrative defaults.
Any change of mandate must be explicit and cannot be mixed into a like-for-like
superiority claim.

An independent data custodian or permission-restricted acquisition job should
materialize and hash each future test snapshot without exposing its values to
the model-development process. The custodian can provide the path, date range,
schema, and SHA-256 commitment. The orchestrator treats its first read—even a
hash verification—as test access and writes the access ledger first.

Validate the preregistration structure. This command does not read artifact
paths or test values:

```bash
python publication_pipeline_draft/future_confirmatory_protocol.py validate \
  --contract frozen_releases/future_confirmatory_v1.json
```

Do not run ad-hoc summaries, plots, correlations, or trial evaluations on the
future test CSV before the locked execution.

## 2. Freeze selection and environment artifacts

The config inventory must include two different objects:

- the complete fixed hyperparameter/architecture/benchmark contract; and
- a selection manifest showing the validation-only decision rule and selected
  model identifiers.

Each model entry records `trained_through` and `selected_through`. Both dates
must be no later than the validation end and strictly before test start.

The environment inventory must contain at least:

- a fully resolved Python lock, preferably `conda-lock.yml` with explicit build
  strings and hashes;
- `renv.lock` plus the repository used to restore it; and
- a hashed runtime-attestation JSON containing exact Python version, R version,
  platform, and immutable container image digest.

Example attestation:

```json
{
  "python_version": "3.13.13",
  "r_version": "4.5.1",
  "platform": "linux-64",
  "container_image_digest": "sha256:REPLACE_WITH_64_HEX_CHARACTERS"
}
```

The test runner itself is a code artifact. It must be able to generate, for
every strategy and preregistered period:

- target weights;
- drifted pretrade weights;
- asset-level realized gross returns;
- actual calendar days;
- gross and net portfolio returns;
- turnover, transaction cost, financing cost, short notional, cash borrowing,
  net exposure, and gross exposure; and
- a complete solver audit for each numerical benchmark.

These fields allow independent recomputation. A self-reported wealth series or
optimizer success message is not an adequate audit.

## 3. Execute exactly once

Use a new output, bundle, and access-ledger path that do not exist:

```bash
LC_ALL=C LANG=C LANGUAGE=C TZ=UTC \
python publication_pipeline_draft/future_confirmatory_protocol.py execute \
  --contract frozen_releases/future_confirmatory_v1.json \
  --repo-root "$PWD" \
  --output immutable_results/future_confirmatory_v1 \
  --bundle immutable_results/future_confirmatory_v1.tar.gz
```

The executor performs the following sequence:

1. schema validation without test access;
2. atomic creation of the external access ledger;
3. hash verification of code, config, data, model, and environment inventories;
4. runtime-attestation and fixed test-calendar verification;
5. locked commands with path inputs supplied only through declared artifact
   placeholders;
6. independent accounting validation, including drift-aware turnover and
   day-prorated financing;
7. solver-audit validation with code 5 rejected; and
8. immutable success or failure archival plus a deterministic tarball checksum.

If any step after ledger creation fails, preserve and report the failure. Fixes
belong to a newly preregistered protocol on later unseen observations.

## 4. Confirmatory inference and reporting

The first inferential table must report the preregistered CRRA-utility effect
against the primary benchmark, a paired dependence-aware interval, the exact
one-sided p-value, and the decision rule. Secondary comparisons use the frozen
multiplicity correction. Report raw monthly paired effects and event counts.

Tail VaR/CVaR remains descriptive until the pooled future sample has at least
the preregistered number of tail events. Do not convert one or two worst months
into a general tail-superiority claim. Report both full-period and partial-period
results, with actual time scaling for partial periods.

Always distinguish:

- market-path inference from paired future returns;
- optimization variability across training seeds; and
- sensitivity analysis, which is not a second confirmatory test.

## Current policy-visible-vine-state and component-ablation inventory

The repository already contains a matched-capacity negative control for the
policy-visible vine state:

- `config/no_vine_ablation_seeds.yaml` preregisters ten distinct zero-vine
  seeds.
- `rl/run_seed_sweep.r` propagates `VINE_OBSERVATION_MODE=zero` through the
  training and sanity workflow.
- `rl/rl_environment.r` zeros the explicit vine state and scenario-CVaR
  observation while retaining the common vine-scenario CVaR reward penalty.
- `rl/training_sanity_check.r`, `rl/train_rl.r`, and
  `rl/policy_inference_server.py` record and validate the explicit no-vine
  signal mask.
- `freeze_training_release.py` and `freeze_evaluation_release.py` can freeze a
  distinct ten-seed policy-state-ablation release.
- `locked_evaluation_batch.py` can evaluate that optional release and label its
  policies as an ablation.
- `eval/ablation.r` is now an artifact-only reader: it requires completed logs
  on identical realized dates and has no synthetic metric fallback.

This isolates policy access to the vine signal; it does **not** remove all vine
machinery or identify the total vine contribution because the reward still uses
vine-scenario CVaR. That broader question needs a separate preregistered
state-and-reward ablation. This is useful machinery, but it is not yet a
completed component-ablation experiment. The successful v4 main archive
records zero such policies. The repository also lacks one preregistered
coordinator for the other required
variants listed in `BENCHMARK_SPECIFICATION.md`:

- no synthetic pretraining;
- matched-capacity feed-forward encoder;
- zero CVaR/risk penalty;
- unlevered long-only mandate;
- static rather than dynamic vine state; and
- alternative monthly scenario generator.

## Missing ablation orchestration to add before publication

A future component-ablation release should add, in new versioned files:

1. one variant manifest defining the exact changed component, invariant model
   capacity where applicable, update budget, seeds, gates, and expected hashes;
2. a batch trainer that refuses partially successful seed sets and emits a
   separate immutable training release per variant;
3. validators for matched parameter count, optimizer-update count, realized
   dates, costs, asset order, and observation masking;
4. one future test coordinator that evaluates every accepted variant on the
   same non-overlapping panel through the common accounting contract;
5. automatic generation of the `config/ablation_manifest.csv`-equivalent from
   frozen weight logs rather than hand-authored result paths; and
6. paired full-model-minus-ablation market effects with multiplicity control.

Do not use the standard error across seeds as market-path evidence. Seed
dispersion belongs in an optimization-robustness appendix; component-value
inference must remain paired on the common future return path.

## Verification command

The new protocol tests are self-contained and do not open the consumed v4
archive:

```bash
python -m pytest -q \
  publication_pipeline_draft/tests/test_future_confirmatory_protocol.py
```

They cover holdout/window overlap, chronology, forbidden validation-data access,
separate artifact hashes, runtime locks, NLopt code 5 rejection, drift-aware
turnover, partial-period financing, immutable success, archived failure, and
single-access enforcement.
