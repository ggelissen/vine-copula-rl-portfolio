# Terminal masked-pretraining controls v1

## Scientific role

This is the final authorized same-holdout neural training experiment. It closes
one material confound left by the synthetic-dose studies: the strongest
100-unique/1000-presentation policy used a masked observation architecture,
whereas the earlier historical-prefix and moving-block-bootstrap controls used
the full policy-visible dependence state.

The experiment trains exactly two controls with the identical TD3-LSTM
architecture, exploration schedule, 1,000 episode presentations, 61 historical
fine-tuning episodes, ten seeds, portfolio constraints, costs, and reward:

1. balanced historical-prefix trajectory repetition;
2. six-month circular moving-block bootstrap pretraining.

The frozen 100-path/1,000-presentation masked NN-vine policy is reused, not
retrained. The two primary contrasts estimate its annualized CRRA certainty-
equivalent difference relative to each control using the common realized path
and paired circular moving-block bootstrap. Holm adjustment covers the two
generator-value hypotheses. Seed effects describe optimization robustness only.

This is post-holdout explanatory evidence. It cannot produce a fresh
confirmatory superiority claim.

## Stop rule

After this experiment, no further model selection or same-holdout neural
training is authorized. Regardless of sign, the computational conclusion is:

- positive versus both controls: conditional evidence that concentrated
  NN-vine pretraining adds value within the masked architecture;
- mixed: generator value is control-dependent and not generally established;
- null/adverse: the observed performance does not require NN-vine pretraining.

Further confirmation requires a genuinely future period or independent panel.
Cost, leverage, ensemble-size, and reporting sensitivity should use frozen
weights and do not require GPU retraining.

## Required immutable inputs

- `data/ablation_training_bundles/historical_prefix_repeated.RData`
- `data/ablation_training_bundles/moving_block_bootstrap.RData`
- `analysis_outputs/synthetic_presentation_response_v2_weights/synthetic_presentation_policy_weight_manifest.csv`
- `protocol_manifests/training_python_runtime.json`
- the frozen realized, causal, and benchmark panels used by v2

The launcher validates their registered SHA-256 hashes before freezing.

## Execution

```bash
cd /gabirel/copula-portfolio-clean
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC

bash hpc/run_masked_pretraining_controls_v1.sh validate | \
  tee logs/masked_pretraining_controls_v1_validation.log
bash hpc/run_masked_pretraining_controls_v1.sh inputs | \
  tee logs/masked_pretraining_controls_v1_inputs.log
bash hpc/run_masked_pretraining_controls_v1.sh freeze | \
  tee logs/masked_pretraining_controls_v1_freeze.log

CONTROL_GPUS=0,1,2,3 CONTROL_CPU_CORES=120 \
  bash hpc/run_masked_pretraining_controls_v1.sh train
bash hpc/run_masked_pretraining_controls_v1.sh status
```

Adjust only `CONTROL_GPUS` and `CONTROL_CPU_CORES` to the allocated hardware.
Do not change the scientific contract, seeds, bundles, or training settings.

After all 20 jobs pass:

```bash
bash hpc/run_masked_pretraining_controls_v1.sh audit | \
  tee logs/masked_pretraining_controls_v1_audit.log
bash hpc/run_masked_pretraining_controls_v1.sh replay | \
  tee logs/masked_pretraining_controls_v1_replay.log
bash hpc/run_masked_pretraining_controls_v1.sh analyze | \
  tee logs/masked_pretraining_controls_v1_analysis.log
bash hpc/run_masked_pretraining_controls_v1.sh checkpoint-archive | \
  tee logs/masked_pretraining_controls_v1_checkpoint_archive.log
bash hpc/run_masked_pretraining_controls_v1.sh finalize | \
  tee logs/masked_pretraining_controls_v1_finalize.log
```

Download both final archives and their sidecars:

- `masked_pretraining_controls_v1_final.tar.gz{,.sha256}`
- `masked_pretraining_controls_v1_checkpoints.tar.gz{,.sha256}`

The first contains publication analysis artifacts. The second preserves the
audited checkpoints and training diagnostics after HPC access ends.
