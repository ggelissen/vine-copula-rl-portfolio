# Exact 60-job causal sweep recovery

This protocol carries forward the exact 70 successful v2 experiment/seed jobs
and retries only the exact 60 failures under v3.  It is acceptable only because
no causal evaluation return was accessed, every retained v2 policy passed the
stricter original gate, and the merger fails on seed or scientific-setting
changes.  The resulting evidence remains post-holdout explanatory.

## 1. Validate and preserve v2

```bash
cd /gabirel/copula-portfolio-clean
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
export PYTHON=/gabirel/miniforge3/bin/python3
export RSCRIPT=/gabirel/miniforge3/bin/Rscript
export TRAIN_PYTHON=/gabirel/miniforge3/envs/vine-rl/bin/python
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python

bash hpc/validate_publication_extension_v2.sh \
  | tee logs/publication_extension_v3_validation.log

test -f protocol_manifests/causal_jobs_v2.csv
test -f protocol_manifests/causal_sweep_status_v2.csv
test -d frozen_releases/publication_extension_v2
test -d logs/publication_extension_v2

tar -czf publication_extension_v2_failed_training_evidence.tar.gz \
  protocol_manifests/causal_jobs_v2.csv \
  protocol_manifests/causal_sweep_status_v2.csv \
  logs/publication_extension_v2 \
  logs/publication_extension_v2.launch.log
sha256sum publication_extension_v2_failed_training_evidence.tar.gz \
  > publication_extension_v2_failed_training_evidence.tar.gz.sha256
```

## 2. Materialize and freeze the full v3 contract

The frozen job contract still contains all 130 keys.  Execution below selects
only the six failed experiment families.

```bash
test ! -e data/publication_extension_runs_v3
test ! -e protocol_manifests/causal_jobs_v3.csv

"$PYTHON" publication_pipeline_draft/causal_ablation_protocol.py \
  --output-root data/publication_extension_runs_v3 \
  --output protocol_manifests/causal_jobs_v3.csv

test "$(($(wc -l < protocol_manifests/causal_jobs_v3.csv) - 1))" -eq 130
(
  cd protocol_manifests
  sha256sum -c causal_jobs_v3.csv.sha256
)

export EXPECTED_TRAIN_GPUS=4
export CAPTURE_TIMING=prospective_before_publication_extension_v3_60_job_retry
export ENV_MANIFEST_DIR="$PWD/provenance_environment_extension_v3_retry"
test ! -e "$ENV_MANIFEST_DIR"
bash hpc/capture_publication_environment.sh

"$PYTHON" publication_pipeline_draft/freeze_publication_extension.py \
  --repo-root . \
  --jobs protocol_manifests/causal_jobs_v3.csv \
  --runtime provenance_environment_extension_v3_retry \
  --bundle-manifest frozen_releases/publication_extension_v2/ablation_bundle_manifest.csv \
  --training-output-root data/publication_extension_runs_v3 \
  --output frozen_releases/publication_extension_v3_retry \
  --archive frozen_releases/publication_extension_v3_retry.tar.gz

(cd frozen_releases/publication_extension_v3_retry && sha256sum -c CONTENTS.sha256)
(cd frozen_releases && sha256sum -c publication_extension_v3_retry.tar.gz.sha256)
```

The frozen v2 bundle manifest is intentionally authoritative for the mixed
70/60 recovery.  This proves that the retry uses the same immutable alternative
pretraining inputs as the 70 carried v2 policies, even if a later Git checkout
replaces the convenience copy under `data/ablation_training_bundles`.

## 3. Launch exactly 60 jobs on four A100s

```bash
FAILED_EXPERIMENTS="historical_only_no_synthetic_pretraining,moving_block_bootstrap_pretraining,feedforward_capacity_matched,sac_lstm_full,ppo_lstm_full,a2c_lstm_full"

test ! -e logs/publication_extension_v3_retry60
test ! -e protocol_manifests/causal_sweep_status_v3_retry60.csv

nohup "$PYTHON" publication_pipeline_draft/run_causal_sweep.py \
  --jobs protocol_manifests/causal_jobs_v3.csv \
  --release frozen_releases/publication_extension_v3_retry \
  --repo-root . \
  --config config/config.yaml \
  --train-python "$TRAIN_PYTHON" \
  --rscript "$RSCRIPT" \
  --gpus 0,1,2,3 \
  --cpu-cores 120 \
  --experiments "$FAILED_EXPERIMENTS" \
  --log-root logs/publication_extension_v3_retry60 \
  --status protocol_manifests/causal_sweep_status_v3_retry60.csv \
  > logs/publication_extension_v3_retry60.launch.log 2>&1 &

echo $! > logs/publication_extension_v3_retry60.pid
```

Monitor:

```bash
PID="$(cat logs/publication_extension_v3_retry60.pid)"
ps -p "$PID" -o pid,etime,stat,cmd
tail -n 30 logs/publication_extension_v3_retry60.launch.log
find data/publication_extension_runs_v3 -name '*_full.pt' | wc -l
nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
  --format=csv,noheader
```

Expected status cardinality after completion:

```bash
"$PYTHON" - <<'PY'
import csv
from pathlib import Path
p = Path("protocol_manifests/causal_sweep_status_v3_retry60.csv")
rows = list(csv.DictReader(p.open()))
assert len(rows) == 60
assert all(row["passed"].lower() == "true" for row in rows)
assert len({(row["experiment_id"], row["seed"]) for row in rows}) == 60
print("Exact 60-job retry passed.")
PY
```

## 4. Merge 70 v2 survivors with 60 v3 retries

```bash
"$PYTHON" publication_pipeline_draft/merge_causal_operational_retry.py \
  --repo-root . \
  --original-jobs protocol_manifests/causal_jobs_v2.csv \
  --original-status protocol_manifests/causal_sweep_status_v2.csv \
  --original-release frozen_releases/publication_extension_v2 \
  --retry-jobs protocol_manifests/causal_jobs_v3.csv \
  --retry-status protocol_manifests/causal_sweep_status_v3_retry60.csv \
  --retry-release frozen_releases/publication_extension_v3_retry \
  --output-jobs protocol_manifests/causal_jobs_v2_v3_merged.csv \
  --output-status protocol_manifests/causal_sweep_status_v2_v3_merged.csv \
  --output-manifest protocol_manifests/causal_v2_v3_operational_merge.json
```

The merger verifies the immutable releases, exact 70/60 split, same 130 keys,
strict-gate success of all carried v2 jobs, complete v3 retries, artifact
existence, and equality of all scientific runtime settings.

## 5. Audit the combined 130 checkpoints

```bash
"$TRAIN_PYTHON" publication_pipeline_draft/audit_causal_sweep.py \
  --jobs protocol_manifests/causal_jobs_v2_v3_merged.csv \
  --status protocol_manifests/causal_sweep_status_v2_v3_merged.csv \
  --operational-merge-manifest protocol_manifests/causal_v2_v3_operational_merge.json \
  --repo-root . \
  --output analysis_outputs/causal_sweep_audit_v2_v3_merged

python3 -m json.tool \
  analysis_outputs/causal_sweep_audit_v2_v3_merged/causal_sweep_audit_manifest.json
```

Required values are `job_count=130`, `mixed_revision_carry_forward=true`,
`v2_carried_count=70`, `v3_retry_count=60`, and
`all_behavior_gate_enforcement_valid=true`.  Economic diagnostic failures may
remain and must later be reported.
