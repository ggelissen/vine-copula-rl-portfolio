# Publication extension v3 recovery

This is an outcome-disclosed operational/scientific revision after the first
130-job training attempt completed 70 jobs.  No causal holdout evaluation was
run.  Preserve the v2 status and logs permanently; never overwrite or relabel
them as v3 evidence.

The revision fixes four diagnosed defects:

1. PPO/A2C terminal masks were Boolean tensors used in arithmetic.
2. The feed-forward capacity search skipped the valid integer width.
3. SAC omitted the shared soft leverage regularizer.
4. Economic behavior gates selected controls differentially before causal
   evaluation.  In v3 they are intent-to-train diagnostics.  Non-finite values,
   action-projection inconsistencies, and hard-constraint violations remain
   fatal.

Two transparent recovery paths are supported.  A complete 130-job v3 rerun is
the single-revision design.  The time-saving design in
`PUBLICATION_EXTENSION_60_JOB_RETRY.md` carries the exact 70 strict-gate v2
successes and reruns the exact 60 failures, with both source releases and the
70/60 merger permanently attested.  Never combine revisions without that
merger and mixed-provenance disclosure.

## 1. Validate the synchronized source

```bash
cd /gabirel/copula-portfolio-clean
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
export PYTHON=/gabirel/miniforge3/bin/python3
export RSCRIPT=/gabirel/miniforge3/bin/Rscript
export TRAIN_PYTHON=/gabirel/miniforge3/envs/vine-rl/bin/python
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python

bash hpc/validate_publication_extension_v2.sh \
  | tee logs/publication_extension_v3_validation.log
```

The server-side suite must exercise the publication-size MLP match and a full
PPO/A2C episode, not only construct the agents.

## 2. Preserve the failed v2 evidence

```bash
test -f protocol_manifests/causal_sweep_status_v2.csv
test -d logs/publication_extension_v2
tar -czf publication_extension_v2_failed_training_evidence.tar.gz \
  protocol_manifests/causal_sweep_status_v2.csv \
  logs/publication_extension_v2 \
  logs/publication_extension_v2.launch.log
sha256sum publication_extension_v2_failed_training_evidence.tar.gz \
  > publication_extension_v2_failed_training_evidence.tar.gz.sha256
```

## 3. Materialize an entirely new v3 job matrix

```bash
test ! -e data/publication_extension_runs_v3
test ! -e protocol_manifests/causal_jobs_v3.csv

"$PYTHON" publication_pipeline_draft/causal_ablation_protocol.py \
  --output-root data/publication_extension_runs_v3 \
  --output protocol_manifests/causal_jobs_v3.csv

test "$(($(wc -l < protocol_manifests/causal_jobs_v3.csv) - 1))" -eq 130
sha256sum -c protocol_manifests/causal_jobs_v3.csv.sha256
```

## 4. Capture and freeze the v3 runtime/code contract

Set `EXPECTED_TRAIN_GPUS` to the GPUs actually allocated to this server.

```bash
export EXPECTED_TRAIN_GPUS=2
export CAPTURE_TIMING=prospective_before_publication_extension_v3
export ENV_MANIFEST_DIR="$PWD/provenance_environment_extension_v3"
test ! -e "$ENV_MANIFEST_DIR"
bash hpc/capture_publication_environment.sh

"$PYTHON" publication_pipeline_draft/freeze_publication_extension.py \
  --repo-root . \
  --jobs protocol_manifests/causal_jobs_v3.csv \
  --runtime provenance_environment_extension_v3 \
  --bundle-manifest data/ablation_training_bundles/ablation_bundle_manifest.csv \
  --training-output-root data/publication_extension_runs_v3 \
  --output frozen_releases/publication_extension_v3 \
  --archive frozen_releases/publication_extension_v3.tar.gz

(cd frozen_releases/publication_extension_v3 && sha256sum -c CONTENTS.sha256)
(cd frozen_releases && sha256sum -c publication_extension_v3.tar.gz.sha256)
```

## 5. Launch the full v3 sweep

For two A100s and 95 CPU cores:

```bash
test ! -e logs/publication_extension_v3
test ! -e protocol_manifests/causal_sweep_status_v3.csv

nohup "$PYTHON" publication_pipeline_draft/run_causal_sweep.py \
  --jobs protocol_manifests/causal_jobs_v3.csv \
  --release frozen_releases/publication_extension_v3 \
  --repo-root . \
  --config config/config.yaml \
  --train-python "$TRAIN_PYTHON" \
  --rscript "$RSCRIPT" \
  --gpus 0,1 \
  --cpu-cores 95 \
  --log-root logs/publication_extension_v3 \
  --status protocol_manifests/causal_sweep_status_v3.csv \
  > logs/publication_extension_v3.launch.log 2>&1 &

echo $! > logs/publication_extension_v3.pid
```

Monitor without changing any run files:

```bash
PID="$(cat logs/publication_extension_v3.pid)"
ps -p "$PID" -o pid,etime,stat,cmd
tail -n 30 logs/publication_extension_v3.launch.log
find data/publication_extension_runs_v3 -name '*_full.pt' | wc -l
nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
  --format=csv,noheader
```

## 6. Audit after completion

```bash
"$TRAIN_PYTHON" publication_pipeline_draft/audit_causal_sweep.py \
  --jobs protocol_manifests/causal_jobs_v3.csv \
  --status protocol_manifests/causal_sweep_status_v3.csv \
  --repo-root . \
  --output analysis_outputs/causal_sweep_audit_v3

python3 -m json.tool \
  analysis_outputs/causal_sweep_audit_v3/causal_sweep_audit_manifest.json
```

`all_behavior_gate_enforcement_valid` must be true.  Economic gate failures
may be present and must be reported; `all_behavior_gates_pass` is no longer a
required truth value.  Do not begin causal holdout replay until all 130 jobs,
checkpoint tensors, metadata, and hard constraints pass this audit.
