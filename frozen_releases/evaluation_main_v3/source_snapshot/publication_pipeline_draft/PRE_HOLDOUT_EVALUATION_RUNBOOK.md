# Pre-holdout evaluation runbook

This sequence freezes and executes the confirmatory full-model comparison
before the computationally expensive secondary ablations. Commands assume the repository root is
`/gabirel/copula-portfolio-clean` and the frozen 20-seed release is
`frozen_releases/training_schema5_v1`.

Evaluation release v3 is an operational revision after three archived attempts
failed before producing policy weights: R Lantern and Python PyTorch exported
incompatible libtorch symbols in one process. The v3 evaluator runs the
unchanged Python actor in a persistent subprocess and exchanges numeric states
and actions with R. No policy, checkpoint, projection, benchmark, cost,
constraint, outcome, or inferential setting changed.

## 1. Verify the new code without opening the holdout

```bash
cd /gabirel/copula-portfolio-clean
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC

python3 -m py_compile \
  publication_pipeline_draft/freeze_evaluation_release.py \
  publication_pipeline_draft/locked_evaluation_batch.py
python3 -m pytest -q publication_pipeline_draft/tests
Rscript --vanilla tests/run_tests.r
Rscript --vanilla tests/test_publication_benchmarks.r
POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python \
/gabirel/miniforge3/bin/Rscript --vanilla \
  tests/test_policy_process_isolation.r
```

Run one complete development-only evaluation using the final 24 training
periods as a mechanical pseudo-holdout. This exercises NN-vine construction,
the actual environment constructor, checkpoint inference, all 24 actions, and
weight export without scoring the locked evaluation window.

```bash
DRY_RUN_DIR="$(mktemp -d /tmp/copula-eval-dry-run.XXXXXX)"
POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python \
EVAL_MODEL_DIR=frozen_releases/training_schema5_v1/seeds/seed_20260741 \
EVAL_OUTPUT_DIR="$DRY_RUN_DIR" \
EVAL_WEIGHTS_ONLY=true \
EVAL_DEVELOPMENT_DRY_RUN=true \
EVAL_WINDOW_ID=development_preflight_v3 \
/gabirel/miniforge3/bin/Rscript --vanilla \
  evaluate_with_config.r config/config.yaml

DRY_RUN_DIR="$DRY_RUN_DIR" /gabirel/miniforge3/bin/python3 - <<'PY'
import os
from pathlib import Path
import pandas as pd

path = Path(os.environ["DRY_RUN_DIR"]) / "weights_rl_full_seed_20260741.csv"
weights = pd.read_csv(path)
columns = [name for name in weights if name.startswith("w_")]
assert len(weights) == 24
assert len(columns) == 7
assert ((weights[columns].sum(axis=1) - 1.0).abs() <= 1e-6).all()
assert (weights[columns].abs().sum(axis=1) <= 1.5 + 1e-6).all()
print("Complete 24-period development evaluation preflight passed:", path)
PY
```

The R tests use only the historical training prefix. The fast DCC test uses a
deterministic covariance provider to test calendar and solver plumbing.

## 2. Run real-method pre-holdout tests

This exercises real `rmgarch` DCC plus static, rolling, and dynamic NN-vine
generation twice. It also perturbs future training-prefix data to test
causality.

```bash
BENCHMARK_TEST_PERIODS=24 \
RUN_EXPENSIVE_BENCHMARK_TESTS=true \
Rscript --vanilla tests/test_publication_benchmarks.r
```

This deliberately uses 24 development decisions ending before the holdout.
Do not freeze or evaluate if this command fails. There is no fallback method.

## 3. Freeze the main evaluation code and contracts

The main release contains all 20 frozen full-model policies and all six causal
benchmarks. It deliberately contains no ablation checkpoint.

```bash
MODE_COUNT="$(find frozen_releases/training_schema5_v1/seeds \
  -name vine_observation_mode.txt | wc -l)"
test "$MODE_COUNT" -eq 0 -o "$MODE_COUNT" -eq 20
if test "$MODE_COUNT" -eq 20; then
  test -z "$(grep -L '^full$' \
    frozen_releases/training_schema5_v1/seeds/*/vine_observation_mode.txt)"
fi

python3 publication_pipeline_draft/freeze_evaluation_release.py \
  --repo-root . \
  --full-training-release frozen_releases/training_schema5_v1 \
  --evaluation-contract publication_pipeline_draft/config/evaluation_contract.json \
  --benchmark-contract publication_pipeline_draft/config/benchmark_contract.json \
  --output frozen_releases/evaluation_main_v3 \
  --bundle frozen_releases/evaluation_main_v3.tar.gz

(cd frozen_releases && sha256sum -c evaluation_main_v3.tar.gz.sha256)
(cd frozen_releases/evaluation_main_v3 && sha256sum -c CONTENTS.sha256)
python3 -m json.tool frozen_releases/evaluation_main_v3/evaluation_release_manifest.json
```

The manifest must say `holdout_accessed_by_freezer=false`, contain 20 full
policies, zero no-vine policies, `secondary_ablations_included=false`, and all
six benchmark IDs.

## 4. Execute the main locked batch exactly once

This is the first command authorized to read and score the final 24 months.
Success or failure is archived immutably.

```bash
POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python \
/gabirel/miniforge3/bin/python3 \
  publication_pipeline_draft/locked_evaluation_batch.py \
  --repo-root . \
  --evaluation-release frozen_releases/evaluation_main_v3 \
  --output locked_evaluation/main_oos_v3_operational_retry \
  --bundle locked_evaluation/main_oos_v3_operational_retry.tar.gz \
  --rscript /gabirel/miniforge3/bin/Rscript

(cd locked_evaluation && sha256sum -c main_oos_v3_operational_retry.tar.gz.sha256)
python3 -m json.tool locked_evaluation/main_oos_v3_operational_retry/locked_batch_manifest.json
```

Only after the bundle and sidecar hash exist should tables and plots under
`locked_evaluation/main_oos_v3_operational_retry/publication_results` be inspected.
The confirmatory decision is recorded in
`publication_results/tables/primary_superiority_decision.json`: a positive
mean CRRA-utility difference versus equal weight and a one-sided paired
moving-block-bootstrap p-value no larger than 0.05 are both required.

## 5. Deferred secondary experiment: matched-capacity no-vine TD3

This creates a separate 10-seed tree and never writes into `data/rl_runs`.

```bash
SWEEP_SEEDS_FILE=config/no_vine_ablation_seeds.yaml \
SWEEP_ROOT_DIR=data/no_vine_rl_runs \
VINE_OBSERVATION_MODE=zero \
Rscript --vanilla rl/run_seed_sweep.r config/config.yaml
```

Confirm the status file contains ten passing rows:

```bash
python3 - <<'PY'
import pandas as pd
p = pd.read_csv('data/no_vine_rl_runs/seed_sweep_status.csv')
assert len(p) == p.seed.nunique() == 10
assert set(p.vine_observation_mode) == {'zero'}
assert set(p.no_vine_signal_mask) == {'explicit_vine_and_scenario_cvar_v1'}
assert (p.training_status == 0).all()
assert (p.sanity_status == 0).all()
assert p.no_holdout_gate_pass.astype(bool).all()
assert p.full_zero_vine_median_action_l1.abs().max() <= 1e-8
print(p[['seed', 'no_holdout_gate_pass']].to_string(index=False))
PY
```

The final assertion is the defining negative-control check: zeroing an already
zero vine channel must not change the no-vine policy.  A positive value means
the full-vine policy was trained accidentally and the ablation is invalid.

## 6. Aggregate and freeze the no-vine release

```bash
python3 publication_pipeline_draft/diagnostic_artifacts.py \
  --rl-runs data/no_vine_rl_runs \
  --expected-seeds 10 \
  --output data/publication_no_vine_training_artifacts_10seeds

tar -czf no_vine_training_artifacts_10seeds.tar.gz \
  -C data publication_no_vine_training_artifacts_10seeds

python3 publication_pipeline_draft/freeze_training_release.py \
  --repo-root . \
  --rl-runs data/no_vine_rl_runs \
  --diagnostics-archive "$PWD/no_vine_training_artifacts_10seeds.tar.gz" \
  --expected-seeds 10 \
  --output frozen_releases/no_vine_schema5_v1 \
  --bundle frozen_releases/no_vine_schema5_v1.tar.gz

sha256sum -c frozen_releases/no_vine_schema5_v1.tar.gz.sha256
(cd frozen_releases/no_vine_schema5_v1 && sha256sum -c CONTENTS.sha256)
```

The no-vine freezer must copy `vine_observation_mode.txt` for every seed. The
evaluation freezer rejects the release unless all ten files contain `zero`.

If this ablation is trained or revised after the main holdout results have been
viewed, label its later comparison secondary/exploratory rather than an
independent confirmatory result. Never change the frozen main result.
