# Mixed pretraining response v1

This is a terminal, post-holdout explanatory experiment. It trains ten new
masked TD3-LSTM policies with 100 frozen NN-vine trajectories and all 61
historical-prefix trajectories proportionally interleaved into exactly 1,000
pretraining presentations. Fine-tuning is the unchanged 61-episode historical
stage. No result from `locked_oos_v1` may be used to modify the design.

The four reported arms are:

1. historical-prefix pretraining plus historical fine-tuning (reused);
2. synthetic pretraining only, evaluated at the pretrained checkpoint (reused);
3. mixed pretraining plus historical fine-tuning (ten new policies);
4. synthetic pretraining plus historical fine-tuning (reused).

All arms use seeds `20261001:20261010`, the selected masked architecture, the
same 1,000-presentation pretraining budget, the same costs and constraints, and
the same realized evaluation panel. Only the mixed arm is newly trained.

## Execution

```bash
cd /gabirel/copula-portfolio-clean
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
export PYTHON=/gabirel/miniforge3/bin/python3
export RSCRIPT=/gabirel/miniforge3/bin/Rscript
export TRAIN_PYTHON=/gabirel/miniforge3/envs/vine-rl/bin/python
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python
export MIXED_GPUS=0,1,2,3
export MIXED_CPU_CORES=80

bash hpc/run_mixed_pretraining_response_v1.sh validate | \
  tee logs/mixed_pretraining_response_v1_validation.log
bash hpc/run_mixed_pretraining_response_v1.sh inputs | \
  tee logs/mixed_pretraining_response_v1_inputs.log
bash hpc/run_mixed_pretraining_response_v1.sh bundle | \
  tee logs/mixed_pretraining_response_v1_bundle.log
bash hpc/run_mixed_pretraining_response_v1.sh freeze | \
  tee logs/mixed_pretraining_response_v1_freeze.log
bash hpc/run_mixed_pretraining_response_v1.sh train
```

Monitor without interrupting the worker:

```bash
bash hpc/run_mixed_pretraining_response_v1.sh status
```

After the status table contains ten passed rows:

```bash
bash hpc/run_mixed_pretraining_response_v1.sh audit | \
  tee logs/mixed_pretraining_response_v1_audit.log
bash hpc/run_mixed_pretraining_response_v1.sh replay | \
  tee logs/mixed_pretraining_response_v1_replay.log
bash hpc/run_mixed_pretraining_response_v1.sh analyze | \
  tee logs/mixed_pretraining_response_v1_analysis.log
bash hpc/run_mixed_pretraining_response_v1.sh checkpoint-archive | \
  tee logs/mixed_pretraining_response_v1_checkpoint_archive.log
bash hpc/run_mixed_pretraining_response_v1.sh finalize | \
  tee logs/mixed_pretraining_response_v1_finalize.log
```

Download both archives and both `.sha256` files. The final results archive
contains the four-arm CSV/LaTeX table, PNG/PDF/TikZ figure, paired block-
bootstrap contrasts, seed-robustness diagnostics, and immutable manifests.
