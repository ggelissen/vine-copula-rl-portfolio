# Publication extension v4 report-only gate recovery

The v3 retry release registered `PRETRAIN_BEHAVIOR_GATE_MODE=report_only`, but
its frozen trainer was an older strict-only implementation.  All 29 failures
were finite economic diagnostics; none was a non-finite diagnostic or hard
constraint failure.  Preserve v3 as disclosed diagnostic evidence and do not
use its checkpoints in the final 70/60 merge.

V4 reruns the same exact 60 v2-failed experiment/seed keys under a release that
statically proves the report-only gate wiring.  The model settings and seeds do
not change.  Economic failures remain reported; `gate_gross_mae`,
`max_position_limit_violation`, and all non-finite diagnostics remain fatal.

The synchronized `rl/train_rl.r` must have SHA-256
`50f056bbfb7a2716eb0436223e3cb044d68544e1bdb744da722d7ceb9d7fd733`
before validation, materialization, or freezing.

## 1. Preserve and validate v3 evidence

```bash
tar -czf publication_extension_v3_retry60_failed_evidence.tar.gz \
  protocol_manifests/causal_jobs_v3.csv \
  protocol_manifests/causal_sweep_status_v3_retry60.csv \
  logs/publication_extension_v3_retry60 \
  logs/publication_extension_v3_retry60.launch.log \
  frozen_releases/publication_extension_v3_retry
sha256sum publication_extension_v3_retry60_failed_evidence.tar.gz \
  > publication_extension_v3_retry60_failed_evidence.tar.gz.sha256

bash hpc/validate_publication_extension_v2.sh \
  | tee logs/publication_extension_v4_validation.log
```

## 2. Materialize and freeze v4

```bash
"$PYTHON" publication_pipeline_draft/causal_ablation_protocol.py \
  --output-root data/publication_extension_runs_v4 \
  --output protocol_manifests/causal_jobs_v4.csv

export EXPECTED_TRAIN_GPUS=4
export CAPTURE_TIMING=prospective_before_publication_extension_v4_gate_recovery
export ENV_MANIFEST_DIR="$PWD/provenance_environment_extension_v4"
test ! -e "$ENV_MANIFEST_DIR"
bash hpc/capture_publication_environment.sh

"$PYTHON" publication_pipeline_draft/freeze_publication_extension.py \
  --repo-root . \
  --jobs protocol_manifests/causal_jobs_v4.csv \
  --runtime provenance_environment_extension_v4 \
  --bundle-manifest frozen_releases/publication_extension_v2/ablation_bundle_manifest.csv \
  --training-output-root data/publication_extension_runs_v4 \
  --output frozen_releases/publication_extension_v4 \
  --archive frozen_releases/publication_extension_v4.tar.gz
```

## 3. Rerun the exact 29 v3 failures

```bash
nohup "$PYTHON" publication_pipeline_draft/run_causal_sweep.py \
  --jobs protocol_manifests/causal_jobs_v4.csv \
  --release frozen_releases/publication_extension_v4 \
  --repo-root . \
  --config config/config.yaml \
  --train-python "$TRAIN_PYTHON" \
  --rscript "$RSCRIPT" \
  --gpus 0,1,2,3 \
  --cpu-cores 120 \
  --retry-failures-from protocol_manifests/causal_sweep_status_v3_retry60.csv \
  --expected-selected-jobs 29 \
  --log-root logs/publication_extension_v4_retry29 \
  --status protocol_manifests/causal_sweep_status_v4_retry29.csv \
  > logs/publication_extension_v4_retry29.launch.log 2>&1 &

echo $! > logs/publication_extension_v4_retry29.pid
```

## 4. Merge and audit only after 60/60 completion

```bash
"$PYTHON" publication_pipeline_draft/merge_causal_three_revision_retry.py \
  --repo-root . \
  --v2-jobs protocol_manifests/causal_jobs_v2.csv \
  --v2-status protocol_manifests/causal_sweep_status_v2.csv \
  --v2-release frozen_releases/publication_extension_v2 \
  --v3-jobs protocol_manifests/causal_jobs_v3.csv \
  --v3-status protocol_manifests/causal_sweep_status_v3_retry60.csv \
  --v3-release frozen_releases/publication_extension_v3_retry \
  --v4-jobs protocol_manifests/causal_jobs_v4.csv \
  --v4-status protocol_manifests/causal_sweep_status_v4_retry29.csv \
  --v4-release frozen_releases/publication_extension_v4 \
  --output-jobs protocol_manifests/causal_jobs_v2_v3_v4_merged.csv \
  --output-status protocol_manifests/causal_sweep_status_v2_v3_v4_merged.csv \
  --output-manifest protocol_manifests/causal_v2_v3_v4_operational_merge.json
```

The merger retains v3 policies only when every strict behavior diagnostic
passed and when all non-gate training sources match v4.  This establishes that
the missing report-only branch could not have changed their trajectory.
