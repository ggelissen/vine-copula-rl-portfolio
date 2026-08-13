# Five-day HPC execution checklist

Use the current server checkout only after syncing the reviewed source changes.
Do not reuse a previously frozen focused release because the focused evaluator,
CE calculation, and financial-benchmark contract changed before execution.

## 1. Validate the synchronized source

```bash
cd /gabirel/copula-portfolio-clean
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
export PYTHON=/gabirel/miniforge3/bin/python3
export RSCRIPT=/gabirel/miniforge3/bin/Rscript
export TRAIN_PYTHON=/gabirel/miniforge3/envs/vine-rl/bin/python
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python

bash hpc/validate_focused_walk_forward_v1.sh | \
  tee logs/focused_walk_forward_v1_validation.log
```

Expected: the focused test and the entire publication test suite pass, all R
protocol tests pass, and process-isolated PyTorch inference passes. Stop on any
skip or failure that is not already explicitly documented.

## 2. Consolidate the completed causal checkpoints

Use the authoritative checkpoint audit from the completed result tree. The
example below assumes the paths used in the final HPC run; adjust only the
audit location or use `--path-remap` if a run tree was moved.

```bash
export CAUSAL_AUDIT=analysis_outputs/causal_checkpoint_audit_v2_v3_v4/checkpoint_audit.csv

"$TRAIN_PYTHON" -m publication_pipeline_draft.freeze_causal_checkpoint_release \
  --repo-root . \
  --audit "$CAUSAL_AUDIT" \
  --output frozen_releases/causal_checkpoints_130_v1 \
  --archive frozen_releases/causal_checkpoints_130_v1.tar.gz

(cd frozen_releases/causal_checkpoints_130_v1 && sha256sum -c CONTENTS.sha256)
(cd frozen_releases && sha256sum -c causal_checkpoints_130_v1.tar.gz.sha256)
```

If audited paths say `data/publication_extension_runs_v3/...` but the live tree
has another root, add for example:

```bash
--path-remap data/publication_extension_runs_v3=/actual/live/v3/root
```

Repeat `--path-remap` for each moved prefix. Copy the archive and sidecar off
HPC and verify the sidecar again before deleting any v2/v3/v4 run directory.

## 3. Freeze and execute the focused study

Follow `publication_pipeline_draft/FOCUSED_WALK_FORWARD_RUNBOOK.md` exactly.
It now defines:

- two non-overlapping 24-month windows;
- three TD3 representations;
- five matched seeds per representation;
- 30 total neural trainings;
- six causal financial benchmarks per window;
- two mechanism contrasts and one separate six-benchmark comparison family;
- target-weight ensembling and common cost/constraint accounting;
- a result archive containing all 30 replay checkpoints.

Do not add a fourth representation, another algorithm, or another window after
viewing outcomes.

## 4. Final off-HPC copy checklist

Before HPC access ends, copy and verify:

- `causal_checkpoints_130_v1.tar.gz` plus sidecar;
- `focused_walk_forward_results_v1.tar.gz` plus sidecar;
- the final causal-results archive plus sidecar;
- the frozen main evaluation archive plus sidecar;
- runtime/environment manifests;
- source commit hash and a source archive;
- final status CSVs and immutable logs.

Only after all copies verify may revision-specific causal runs, failed retries,
generator caches, and window working directories be deleted from HPC.

## 5. Stop conditions

Stop rather than repair scientifically if any of these occurs after result
access: missing matched seed, changed checkpoint hash, altered dates/assets,
solver fallback, hard constraint violation, regenerated input for only one
variant, or post-result edits to the frozen contract. Operational path/runtime
repairs must be documented and must not change the scientific contract.
