# Post-holdout execution runbook

This runbook starts from the completed `main_oos_v4_operational_retry` batch.
That 24-month holdout is consumed. Do not rerun, tune on, or relabel it as a
fresh confirmatory result. The work below either preserves the primary result,
explains mechanisms on the same sample, or prepares a genuinely new sample.

## 1. Validate the revised framework on the Linux host

```bash
cd /gabirel/copula-portfolio-clean
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC

/gabirel/miniforge3/bin/python3 -m compileall -q publication_pipeline_draft
/gabirel/miniforge3/bin/python3 -m pytest -q publication_pipeline_draft/tests
/gabirel/miniforge3/bin/Rscript --vanilla tests/run_tests.r
/gabirel/miniforge3/bin/Rscript --vanilla tests/test_publication_benchmarks.r

POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python \
  /gabirel/miniforge3/bin/Rscript --vanilla \
  tests/test_policy_process_isolation.r
```

Do not set `RUN_EXPENSIVE_BENCHMARK_TESTS=true` against the consumed holdout.
The normal tests are source/protocol checks and do not create a new OOS result.

## 2. Re-verify, but never re-execute, the successful v4 result

```bash
(cd locked_evaluation && \
  sha256sum -c main_oos_v4_operational_retry.tar.gz.sha256)

/gabirel/miniforge3/bin/python3 -m json.tool \
  locked_evaluation/main_oos_v4_operational_retry/locked_batch_manifest.json
```

The manifest must remain `status: complete`, with 20 full policies, six
benchmarks, and no no-vine policies. The archive SHA-256 used by every new
contract is:

```text
770d2944f915d0ad21ae9af32e31d68d652fdb54e98939caeab45c327b4e5ea1
```

## 3. Capture provenance before new expensive work

```bash
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python
export TRAIN_PYTHON=/absolute/path/to/the-discovered-gpu-python
export RSCRIPT=/gabirel/miniforge3/bin/Rscript
export PYTHON=/gabirel/miniforge3/bin/python3
export CONDA=/gabirel/miniforge3/bin/conda

bash hpc/capture_publication_environment.sh
(cd provenance_environment_v4 && sha256sum -c CONTENTS.sha256)
```

This capture is deliberately labelled retrospective. If an immutable image
digest or pre-run environment lock exists, add it to the provenance package;
that evidence is stronger than a retrospective package inventory.

List every successful/failed archive pair and then build the provenance pack:

```bash
find locked_evaluation -maxdepth 1 -type f \
  \( -name '*.tar.gz' -o -name '*.tar.gz.sha256' \) -print | sort
```

Follow `publication_pipeline_draft/PUBLICATION_PROVENANCE_RUNBOOK.md`. Include
one `--failed-retry ARCHIVE SIDECAR` pair for every preserved failed batch,
the exact `evaluation_main_v4` and `training_schema5_v1` directories, the raw
market-data hash/declaration, and the files in `provenance_environment_v4`.

## 4. Freeze the secondary-experiment plan

```bash
/gabirel/miniforge3/bin/python3 \
  publication_pipeline_draft/secondary_experiment_protocol.py freeze \
  --repo-root . \
  --contract publication_pipeline_draft/config/secondary_experiments_v1.json \
  --output frozen_releases/post_holdout_secondary_plan_v5 \
  --bundle frozen_releases/post_holdout_secondary_plan_v5.tar.gz

(cd frozen_releases && \
  sha256sum -c post_holdout_secondary_plan_v5.tar.gz.sha256)
(cd frozen_releases/post_holdout_secondary_plan_v5 && \
  sha256sum -c CONTENTS.sha256)
```

If that output already exists, verify it; do not delete or overwrite it. A
scientific or operational change requires a new versioned contract and output.

Validate the existing 20 paired pretraining/full checkpoints before spending
GPU time:

```bash
/gabirel/miniforge3/bin/python3 \
  publication_pipeline_draft/secondary_experiment_protocol.py \
  validate-checkpoints \
  --contract publication_pipeline_draft/config/secondary_experiments_v1.json \
  --experiment pretrained_only \
  --training-release frozen_releases/training_schema5_v1
```

## 5. Select and prove the GPU training Python

Use the same GPU-enabled Python that successfully trained the full model. Do
not use the CPU-only policy-inference environment. The path below is a
placeholder, not a command that can work literally. First inventory the conda
environments and canonical Python executables:

```bash
/gabirel/miniforge3/bin/conda env list
nvidia-smi -L

find /gabirel/miniforge3 /gabirel/venvs \
  -path '*/bin/python' -executable -print 2>/dev/null | sort -u
```

Test each candidate in its own process. Import failures are reported without
modifying the candidate environment:

```bash
while IFS= read -r candidate; do
  echo "===== $candidate ====="
  timeout 30s "$candidate" -c \
    'import sys, torch; print("python=", sys.executable); print("torch=", torch.__version__); print("torch_path=", torch.__file__); print("cuda_runtime=", torch.version.cuda); print("cuda_available=", torch.cuda.is_available()); print("gpu_count=", torch.cuda.device_count())' \
    2>&1 || echo "candidate_failed"
done < <(
  find /gabirel/miniforge3 /gabirel/venvs \
    -path '*/bin/python' -executable -print 2>/dev/null | sort -u
)
```

Select only a candidate reporting `cuda_available=True` and `gpu_count` of at
least four. Set its exact path, then run:

```bash
export TRAIN_PYTHON=/absolute/path/to/gpu-training-python

"$TRAIN_PYTHON" - <<'PY'
import sys
import torch
print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 4
PY
```

This is the first unavoidable operator-supplied value: the correct path cannot
be inferred safely from the repository because the CPU evaluation environment
and GPU training environment intentionally differ.

## 6. Train the ten-seed policy-visible-vine-state negative control on four GPUs

The launcher gives each worker one A100 and 18 vine-simulation CPU cores, using
72 of the 78 available cores. It refuses an existing output root and a silent
CPU fallback.

```bash
export RSCRIPT=/gabirel/miniforge3/bin/Rscript
export PYTHON=/gabirel/miniforge3/bin/python3
export NO_VINE_SWEEP_ROOT="$PWD/data/no_vine_rl_runs_secondary_v2"
export VINE_SIM_CORES_PER_WORKER=18

bash hpc/run_no_vine_4gpu.sh
```

On completion, validate independently of the launcher:

```bash
/gabirel/miniforge3/bin/python3 \
  publication_pipeline_draft/secondary_experiment_protocol.py \
  validate-sweep \
  --contract publication_pipeline_draft/config/secondary_experiments_v1.json \
  --experiment no_vine_td3 \
  --status data/no_vine_rl_runs_secondary_v2/seed_sweep_status.csv
```

All ten seeds, both training/sanity exit codes, both behavior gates, the zero
signal mask, and zero-channel invariance must pass. Partial success is not
silently accepted.

## 7. Aggregate and freeze the no-vine training release

```bash
/gabirel/miniforge3/bin/python3 \
  publication_pipeline_draft/diagnostic_artifacts.py \
  --rl-runs data/no_vine_rl_runs_secondary_v2 \
  --expected-seeds 10 \
  --output data/publication_no_vine_training_artifacts_10seeds

tar -czf no_vine_training_artifacts_10seeds.tar.gz \
  -C data publication_no_vine_training_artifacts_10seeds

/gabirel/miniforge3/bin/python3 \
  publication_pipeline_draft/freeze_training_release.py \
  --repo-root . \
  --rl-runs data/no_vine_rl_runs_secondary_v2 \
  --diagnostics-archive "$PWD/no_vine_training_artifacts_10seeds.tar.gz" \
  --expected-seeds 10 \
  --output frozen_releases/no_vine_schema5_secondary_v1 \
  --bundle frozen_releases/no_vine_schema5_secondary_v1.tar.gz

(cd frozen_releases && \
  sha256sum -c no_vine_schema5_secondary_v1.tar.gz.sha256)
(cd frozen_releases/no_vine_schema5_secondary_v1 && \
  sha256sum -c CONTENTS.sha256)
```

## 8. Run the same-sample explanatory ablation batch once

This creates two missing comparisons: the existing 20 pretrained-only
checkpoints and the new ten-seed policies without policy-visible vine state.
The common vine-scenario CVaR reward remains active, so this does not estimate
the total contribution of all vine machinery. Realized returns, benchmark
weights, and full-policy weights are copied from v4 byte-for-byte. The batch
replays v4 economics and refuses a mismatch.

```bash
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python

POLICY_PYTHON="$POLICY_PYTHON" \
/gabirel/miniforge3/bin/python3 \
  publication_pipeline_draft/secondary_ablation_batch.py \
  --repo-root . \
  --contract publication_pipeline_draft/config/secondary_evaluation_contract_v1.json \
  --successful-archive locked_evaluation/main_oos_v4_operational_retry.tar.gz \
  --successful-sidecar locked_evaluation/main_oos_v4_operational_retry.tar.gz.sha256 \
  --evaluation-contract publication_pipeline_draft/config/evaluation_contract.json \
  --runtime-config config/config.yaml \
  --full-training-release frozen_releases/training_schema5_v1 \
  --no-vine-training-release frozen_releases/no_vine_schema5_secondary_v1 \
  --secondary-plan-release frozen_releases/post_holdout_secondary_plan_v5 \
  --output secondary_evaluation/post_holdout_explanatory_ablation_v1 \
  --bundle secondary_evaluation/post_holdout_explanatory_ablation_v1.tar.gz \
  --rscript /gabirel/miniforge3/bin/Rscript

(cd secondary_evaluation && \
  sha256sum -c post_holdout_explanatory_ablation_v1.tar.gz.sha256)
(cd secondary_evaluation/post_holdout_explanatory_ablation_v1 && \
  sha256sum -c CONTENTS.sha256)

/gabirel/miniforge3/bin/python3 -m json.tool \
  secondary_evaluation/post_holdout_explanatory_ablation_v1/\
post_holdout_explanatory_release_manifest.json
```

Acceptance requires `release_status` equal to
`frozen_post_holdout_explanatory_ablation`,
`confirmatory_claims_permitted=false`, `economic_replay_verified=true`,
`full_inference_replay_verified=true`, and `matched_design_verified=true`.
Every row of `post_holdout_explanatory_v4_replay_verification.csv` must be
`exact_within_tolerance`.

## 9. Interpretation and next genuinely confirmatory study

The existing ensemble-mechanism package is already available under
`analysis_outputs/oos_v4_verified_770d2944/post_holdout_ensemble_mechanism_v2`.
It explains cancellation and cost savings; it does not repair the mixed primary
result.

After the state-ablation/pretrained batch, report effect sizes and seed distributions
as secondary mechanism evidence. Do not select a redesigned model using these
24 months and then call another score on them confirmatory.

For the next confirmatory claim, copy and edit
`publication_pipeline_draft/config/future_confirmatory_contract.example.json`,
replace its illustrative dates/artifact hashes/runtime locks with a genuinely
non-overlapping future or external-market panel, and validate it with:

```bash
/gabirel/miniforge3/bin/python3 \
  publication_pipeline_draft/future_confirmatory_protocol.py validate \
  --contract publication_pipeline_draft/config/future_confirmatory_contract.json
```

Do not execute the example contract itself: its dates and three-period windows
are structural placeholders, not a powered empirical design.
