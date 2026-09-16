# Neural-vine scenarios for recurrent portfolio control

Research code for **Dynamic Portfolio Optimisation under Tail Risk: Combining Neural Vines with LSTM–TD3 Reinforcement Learning** (Gabriël M. J. Gelissen and Fengyi Yuan; manuscript in preparation). The model combines a time-varying Student-t D-vine scenario generator with constrained, recurrent portfolio control. The preferred *development* specification uses a scalar scenario-CVaR signal, mixed historical and synthetic experience during pretraining, and historical fine-tuning. The original full-vine-state specification is retained as a separately evaluated reference.

## Results and evidence status

The original 20-seed full-state policy was assessed in a frozen out-of-sample comparison with six benchmarks. Its positive CRRA certainty-equivalent difference from equal weight was **not statistically significant** (one-sided block-bootstrap p = 0.2347). The preferred mixed, compressed-state specification was chosen **after** that comparison. On the common 22 complete monthly periods, its ten-seed mean-weight ensemble had 34.13% CAGR, 2.31 Sharpe ratio, 33.07% annual CRRA certainty equivalent, and mean monthly target-weight turnover of 0.351. These are observed, post-selection development results—not a fresh confirmatory test or evidence of universal superiority. Ten matched seeds and block-bootstrap intervals do not establish a significant CE advantage over historical-only training.

The study's theoretical contribution concerns conditional tail-risk information, not a theorem that the learned policy will outperform in markets. Read the manuscript, its tables, and the source evidence together. [The evidence synthesis](publication_pipeline_draft/PROJECT_EVIDENCE_SYNTHESIS.md) describes the original frozen comparison and later explanatory work; its earlier numerical summaries do not describe the final mixed-curriculum model.

## Find the relevant files

| Purpose | Location |
| --- | --- |
| Data loading, chronological splits, marginal models | `helper/` |
| Neural vine and benchmark models | `benchmark_models/` |
| Synthetic episodes, environment, recurrent TD3, policy replay | `rl/` |
| Common accounting and experimental protocols | `publication_pipeline_draft/`, `eval/` |
| Configuration and registered contrasts | `config/`, `publication_pipeline_draft/config/` |
| Cluster workflows and protocol tests | `hpc/`, `tests/`, `publication_pipeline_draft/tests/` |
| Frozen evidence and checksums | `frozen_releases/` |
| Result tables and figure-generation code | `analysis_outputs/`, `publication_pipeline_draft/` |
| Dataset provenance and redistribution limits | [data/README.md](data/README.md) |

Historical analyses and operational runbooks remain for traceability. A directory ending in `v1`, `v2`, etc. does **not** mean those versions are interchangeable scientific results. Consult the release manifest and checksum before using an artifact.

## Reproduction

The full pipeline requires R, Python, and substantial CPU/GPU resources. Package versions and commands depend on the experiment; a smoke test is not a reproduction of the paper. Begin with the relevant runbook and frozen contract, supply authorised source data as described in [data/README.md](data/README.md), and verify release checksums before scoring.

Fast, non-training checks from the repository root:

```bash
Rscript --vanilla tests/run_tests.r
Rscript --vanilla tests/test_publication_benchmarks.r
python3 -m pytest -q publication_pipeline_draft/tests
```

The primary historical evaluation is described in [the pre-holdout evaluation runbook](publication_pipeline_draft/PRE_HOLDOUT_EVALUATION_RUNBOOK.md). Later mixed-training evidence and its publication artifacts are described in [the mixed-pretraining runbook](publication_pipeline_draft/MIXED_PRETRAINING_PUBLICATION_RUNBOOK.md). **Do not retune on the consumed holdout and present the result as new confirmation.** A new confirmatory claim needs a separately frozen future or external-market test.

## Data, artifacts, and citation

Input prices may be subject to provider redistribution terms; the GitHub source tree alone may be insufficient for byte-for-byte reproduction. Generated episodes, checkpoints, and large releases may be Git-LFS objects or separately archived assets. Small manifests, hashes, source code, and final summary tables should remain accessible without downloading training checkpoints. Each manuscript result should be traceable to its release manifest and exact code commit.

If using the code, cite the software with [CITATION.cff](CITATION.cff). Once the manuscript has a DOI or arXiv identifier, add it here and to the citation metadata. For a code release, cite a version-specific DOI or tag rather than the mutable `main` branch.

## Scope and reuse

This is research software, not an investment product or trading recommendation. A repository license has not yet been selected; public visibility alone does not grant general reuse rights. Dataset rights must be assessed separately from any future code license.
