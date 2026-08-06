# Post-holdout explanatory ablation runbook

This batch is deliberately **not confirmatory**. The `locked_oos_v1` holdout
was consumed by the successful v4 evaluation. The policy-visible-vine-state
and pretrained-only comparisons can explain mechanisms on that same sample;
any confirmatory claim
requires a non-overlapping future or external evaluation window.

## Inputs

- The immutable successful v4 archive and its SHA-256 sidecar.
- The exact evaluation contract used by v4.
- The frozen 20-seed full training release, containing both `full` and
  `pretrained` checkpoints.
- The frozen 10-seed no-policy-visible-vine-state training release, with
  `vine_observation_mode=zero` evidence.
- The frozen secondary-plan source snapshot. Before any checkpoint inference,
  every executable live source byte must match that snapshot; otherwise the
  batch stops.

Legacy v2 no-vine runs omitted the redundant mode field from the manifest and
checkpoint metadata. Recovery is permitted only when the repair script proves
the intervention from the frozen seed specification, hash-matched per-seed
source snapshots for the runner/launcher/trainer/environment chain, and the
immutable worker log for each unique completed seed. It writes an explicit
post-hoc attestation and plaintext marker but never rewrites a checkpoint,
metric, manifest, or prior status. If any link in that evidence chain is absent
or contradictory, the recovery fails closed and the ablation must be retrained.

The orchestrator refuses an archive other than SHA-256
`770d2944f915d0ad21ae9af32e31d68d652fdb54e98939caeab45c327b4e5ea1`.
It copies v4 realized returns, benchmark weights, and full-policy weights
byte-for-byte. It sets `EVAL_CHECKPOINT_MODELS=pretrained` for the existing
full-model pretraining checkpoints and `EVAL_CHECKPOINT_MODELS=full` plus
`VINE_OBSERVATION_MODE=zero` for the state-ablation checkpoints. The common
vine-scenario CVaR reward remains active, so this does not remove the entire
vine mechanism.

## Run once

```bash
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python
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
  --secondary-plan-release frozen_releases/post_holdout_secondary_plan_v6 \
  --output secondary_evaluation/post_holdout_explanatory_ablation_v1 \
  --bundle secondary_evaluation/post_holdout_explanatory_ablation_v1.tar.gz \
  --rscript /gabirel/miniforge3/bin/Rscript

(cd secondary_evaluation && \
  sha256sum -c post_holdout_explanatory_ablation_v1.tar.gz.sha256)
```

Do not delete or overwrite a completed output to obtain a different result. A
corrected execution must use a new explicitly labelled output/version.

## Acceptance checks

1. Verify the emitted bundle sidecar with `sha256sum -c`.
2. Confirm the release manifest says
   `frozen_post_holdout_explanatory_ablation`,
   `confirmatory_claims_permitted=false`, and
   `economic_replay_verified=true`, `full_inference_replay_verified=true`, and
   `matched_design_verified=true`.
3. Confirm every row of
   `post_holdout_explanatory_v4_replay_verification.csv` is
   `exact_within_tolerance`. This proves the realized path, benchmark/full
   weights, and implementation-cost calculation reproduce v4.
4. Treat descriptive differences, including any favorable result, only as
   mechanism evidence. Do not report p-values or same-sample superiority tests
   as confirmatory.
