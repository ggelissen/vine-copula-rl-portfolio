# GPU training on Slurm

`train_rl_gpu_array.sbatch` launches four independent, reproducible training
replicas.  Each array task receives one GPU, eight CPU cores, and 32 GB RAM;
it writes checkpoints and logs under `data/rl_runs/<array-job>_<task>/`.
The replicas must not share an output directory.

Use an A100 80 GB or H200 partition if readily available.  This model needs
little GPU memory, so a single A10/A6000/V100 is also sufficient; newer GPUs
mainly reduce the PyTorch update time.  Start with four one-GPU tasks rather
than multiple GPUs inside one training process: the current R environment is
stateful and does not implement distributed/vectorized rollouts.

Before submission, create an environment containing a CUDA-enabled PyTorch,
NumPy, Gym, and the R installation/packages used by this repository.  Then
generate the corrected episodic bundle once, set its name, and submit:

```bash
Rscript --vanilla rl/synthetic_returns.r config/config.yaml
export RL_CONDA_ENV=vine-rl
sbatch hpc/train_rl_gpu_array.sbatch
```

Check that each task log reports `Device: cuda` and the intended GPU name.
Choose the final checkpoint only after evaluating all replica output folders.
