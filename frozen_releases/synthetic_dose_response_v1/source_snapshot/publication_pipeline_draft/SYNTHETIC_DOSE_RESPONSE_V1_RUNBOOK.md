# Synthetic-dose response v1

## Scientific scope

This is a post-holdout explanatory experiment. The original 24-month holdout
has already been consumed, so a favorable result cannot be described as fresh
confirmation. The experiment tests whether reducing NN-vine simulator exposure
from 1,000 to 100 unique episode presentations mitigates negative transfer.

The earlier historical-prefix control used 1,000 matched episode presentations.
It therefore does not establish that fewer optimizer steps caused its strong
result. Version 1 deliberately tests the practically relevant combined dose
(100 unique paths, each presented once). If it is favorable, a later repeated-
100 update-budget control can distinguish path diversity from update count.

Exactly two representations are trained with the same ten seeds used in the
earlier causal study:

1. full raw NN-vine state plus scenario-CVaR;
2. no policy-visible dependence, with raw vine features and scenario-CVaR
   observations zeroed while dependence-informed synthetic returns and the
   CVaR reward remain active.

This second representation was selected after the completed w01/w02
retrospective analysis ranked it first by CRRA certainty equivalent in both
windows. The selection is therefore explicitly post-result and cannot turn the
dose experiment into confirmatory evidence. The previously proposed
scenario-CVaR-only 100-path arm was removed before freezing or training, keeping
the experiment at 20 jobs while targeting the more informative architecture.

The 100 episodes are selected before evaluation with version-independent
midpoint systematic indices `6, 16, ..., 996`. All 61 historical trajectories
remain in fine-tuning. The random warm-up is reduced from 1,000 to 100 steps and
the noise decay is changed from `0.998` to `0.998^10`; these preserve the
original warm-up fraction and normalized exploration curve across a run that is
one tenth as long.

## Operational rule

Do not synchronize these source changes into an HPC checkout while an existing
multi-job sweep is still starting new R processes. Finish and audit `w02` first,
then pull this revision and run the commands below.

Set the hardware values for the allocation actually available:

```bash
cd /gabirel/copula-portfolio-clean
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
export PYTHON=/gabirel/miniforge3/bin/python3
export RSCRIPT=/gabirel/miniforge3/bin/Rscript
export TRAIN_PYTHON=/gabirel/miniforge3/envs/vine-rl/bin/python
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python
export DOSE_GPUS=0,1,2,3
export DOSE_CPU_CORES=120
```

Validate code, create the immutable 100-path bundle, materialize the job matrix,
and freeze the release:

```bash
bash hpc/run_synthetic_dose_response_v1.sh validate | \
  tee logs/synthetic_dose_response_v1_validation.log
bash hpc/run_synthetic_dose_response_v1.sh bundle | \
  tee logs/synthetic_dose_response_v1_bundle.log
bash hpc/run_synthetic_dose_response_v1.sh freeze | \
  tee logs/synthetic_dose_response_v1_freeze.log
```

Launch the 20 jobs and monitor them:

```bash
bash hpc/run_synthetic_dose_response_v1.sh train
tail -f logs/synthetic_dose_response_v1.launch.log
watch -n 30 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
```

After the status reports 20/20 passed, audit and replay weights:

```bash
bash hpc/run_synthetic_dose_response_v1.sh audit | \
  tee logs/synthetic_dose_response_v1_audit.log
bash hpc/run_synthetic_dose_response_v1.sh replay | \
  tee logs/synthetic_dose_response_v1_replay.log
bash hpc/run_synthetic_dose_response_v1.sh analyze | \
  tee logs/synthetic_dose_response_v1_analysis.log
```

Preserve the status, audit, replay weights, and logs before scoring. The common
accounting comparison must reuse the exact realized panel, dates, asset order,
constraints, turnover convention, transaction costs, and financing costs from
the frozen v4 evaluation and the causal analysis. Ten seeds characterize
optimization variability; they are not ten independent market histories.

The analyzer produces common-accounting seed and ensemble metrics, three
primary dose/representation contrasts, two historical-only comparisons, six
financial-benchmark comparisons, block-bootstrap intervals with separately
controlled Holm families, and two figures. The frozen 22-period complete-sample
analysis remains primary; an automatically generated 24-period locked-all
sensitivity discloses whether the legacy minimum-trading-day flag changes any
conclusion. No result is allowed to change the post-holdout explanatory label.
