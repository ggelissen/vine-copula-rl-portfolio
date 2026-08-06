# Publication provenance and incident evidence

This package documents the already-completed locked OOS experiment. It does
not rerun, replace, rescore, or modify the successful V4 result.

## Required evidence

- the extracted successful batch, its original tarball, and SHA-256 sidecar;
- the exact frozen pre-holdout evaluation release;
- the exact frozen pre-OOS training release;
- every failed locked-retry tarball and sidecar, in chronological order;
- at least one raw-data source, or an explicit licensed/external declaration;
- environment locks or manifests for both R and Python/container execution.

Missing required evidence is fatal. Licensed data may remain external only
when its immutable SHA-256, URI/identifier, and license are declared.

## Build the supplementary evidence package

First capture the current Linux runtime inventory. This is retrospective
evidence and the generated metadata says so explicitly; it is not a substitute
for a pre-run lock or immutable container digest.

```bash
export POLICY_PYTHON=/gabirel/venvs/copula-eval-torch271-cpu/bin/python
export RSCRIPT=/gabirel/miniforge3/bin/Rscript
export PYTHON=/gabirel/miniforge3/bin/python3
export CONDA=/gabirel/miniforge3/bin/conda
bash hpc/capture_publication_environment.sh
(cd provenance_environment_v1 && sha256sum -c CONTENTS.sha256)
```

Inventory every failed locked archive and sidecar before running the packager:

```bash
find locked_evaluation -maxdepth 1 -type f \
  \( -name '*.tar.gz' -o -name '*.tar.gz.sha256' \) -print | sort
```

The example below shows the known v2 and v3 incidents. Add one
`--failed-retry ARCHIVE SIDECAR` pair for every earlier failed archive that is
present; omission is not allowed merely because the failure was operational.

```bash
python3 publication_pipeline_draft/assemble_publication_provenance.py \
  --successful-batch-directory locked_evaluation/main_oos_v4_operational_retry \
  --successful-batch-archive locked_evaluation/main_oos_v4_operational_retry.tar.gz \
  --successful-batch-sidecar locked_evaluation/main_oos_v4_operational_retry.tar.gz.sha256 \
  --evaluation-release frozen_releases/evaluation_main_v4 \
  --training-release frozen_releases/training_schema5_v1 \
  --failed-retry locked_evaluation/main_oos_v2_operational_retry.tar.gz \
                 locked_evaluation/main_oos_v2_operational_retry.tar.gz.sha256 \
  --failed-retry locked_evaluation/main_oos_v3_operational_retry.tar.gz \
                 locked_evaluation/main_oos_v3_operational_retry.tar.gz.sha256 \
  --raw-data market_prices=data/portfolio_B_7assets_2013.csv \
  --environment-manifest provenance_environment_v1/capture_metadata.json \
  --environment-manifest provenance_environment_v1/system_runtime.txt \
  --environment-manifest provenance_environment_v1/r_installed_packages.csv \
  --environment-manifest provenance_environment_v1/conda_explicit_linux-64.txt \
  --environment-manifest provenance_environment_v1/orchestration_python_runtime.json \
  --environment-manifest provenance_environment_v1/policy_python_runtime.json \
  --environment-manifest provenance_environment_v1/policy_python_pip_freeze.txt \
  --output frozen_releases/publication_provenance_v1 \
  --bundle frozen_releases/publication_provenance_v1.tar.gz
```

For licensed data that cannot be redistributed, replace `--raw-data` with:

```bash
--external-raw-data 'market_prices|<64-hex-sha256>|<DOI-or-vendor-URI>|<license>'
```

Raw bytes and checkpoint/large training artifacts are hash-inventoried but not
duplicated by default. Use `--copy-raw-data` and/or
`--copy-large-artifacts` only when redistribution and storage permit it.

## Verify before deposit

```bash
(cd frozen_releases && sha256sum -c publication_provenance_v1.tar.gz.sha256)
(cd frozen_releases/publication_provenance_v1 && sha256sum -c CONTENTS.sha256)
python3 -m json.tool \
  frozen_releases/publication_provenance_v1/validation_report.json
```

The report must have zero required-check failures. Warnings about omitted
large bytes and failed retry incidents are disclosures, not permission to
alter the locked result. Deposit the bundle, sidecar, and all licensed-data
access instructions together. Report the V4 statistical conclusion unchanged.
