# Final synthetic diversity/presentation identification experiment

This post-holdout explanatory experiment holds the exact synthetic path set at
100 unique episodes and restores 1,000 episode presentations through ten
ordered complete passes. It therefore separates two changes that the v1
100-path experiment made simultaneously:

1. **Presentation effect:** repeated-100 (100 unique / 1,000 presentations)
   versus v1 (100 unique / 100 presentations).
2. **Diversity effect:** repeated-100 versus the original causal policies
   (1,000 unique / 1,000 presentations).

Both the full-state and no-policy-visible-dependence TD3-LSTM policies use the
same ten seeds as v1. The 100-path source, historical fine-tuning episodes,
architecture, optimizer, transaction costs, action constraints, evaluation
calendar and realized returns are unchanged. The experiment reuses a consumed
holdout and cannot support a new confirmatory superiority claim.

Use `hpc/run_synthetic_presentation_response_v2.sh` in this order:

```bash
bash hpc/run_synthetic_presentation_response_v2.sh validate
bash hpc/run_synthetic_presentation_response_v2.sh bundle
bash hpc/run_synthetic_presentation_response_v2.sh freeze
bash hpc/run_synthetic_presentation_response_v2.sh train
bash hpc/run_synthetic_presentation_response_v2.sh status
bash hpc/run_synthetic_presentation_response_v2.sh audit
bash hpc/run_synthetic_presentation_response_v2.sh replay
bash hpc/run_synthetic_presentation_response_v2.sh analyze
bash hpc/run_synthetic_presentation_response_v2.sh finalize
```

Set `PRESENTATION_GPUS` and `PRESENTATION_CPU_CORES` before `train` if the HPC
allocation differs from the defaults. Outputs are immutable and commands fail
closed when an output already exists; never delete successful runs to retry a
different scientific specification under the same release identifier.
