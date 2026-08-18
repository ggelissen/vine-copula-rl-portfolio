# Terminal publication artifacts

This pipeline transforms the immutable terminal-robustness campaign into an
additive manuscript bundle. It does not modify the existing TikZ directory or
any manuscript chapter.

## Generate on the HPC repository

```bash
cd /gabirel/copula-portfolio-clean
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
PYTHON=/gabirel/miniforge3/bin/python3

"$PYTHON" publication_pipeline_draft/generate_terminal_publication_artifacts.py \
  --repo-root . \
  --terminal-results analysis_outputs/terminal_robustness_v1
```

If and only if the terminal bundle itself already exists and should be
regenerated, append `--replace`. This replaces only
`manuscript_revision_causal_v1/publication_terminal_v1`; it never replaces the
existing publication figures.

Verify the generated bundle:

```bash
(
  cd manuscript_revision_causal_v1/publication_terminal_v1
  sha256sum -c CONTENTS.sha256
)

"$PYTHON" -m pytest -q \
  publication_pipeline_draft/tests/test_terminal_publication_artifacts.py \
  publication_pipeline_draft/tests/test_publication_tikz.py
```

If a LaTeX installation is available, compile the proof booklet:

```bash
cd manuscript_revision_causal_v1/publication_terminal_v1/figures/tikz
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  preview_terminal_figures.tex
```

The terminal bundle intentionally emits `\pgfplotsset{compat=1.16}` because
the current HPC image uses TeX Live 2019. This override is isolated to the new
terminal proof bundle and does not change the existing manuscript preamble.

## Generate locally in PowerShell

When the terminal archive has been extracted under `analysis_work`, run:

```powershell
cd C:\Users\gabri\Downloads\copula-based_dynamic_portfolio_selection

python .\publication_pipeline_draft\generate_terminal_publication_artifacts.py `
  --repo-root . `
  --terminal-results .\analysis_work\terminal_robustness_v1_review\analysis_outputs\terminal_robustness_v1
```

For later regeneration of this new bundle, add `--replace`.

## Import into the manuscript

The existing manuscript already inputs the shared TikZ preamble, so do not
input the copied terminal preamble a second time. Import a new figure from the
manuscript root as follows:

```tex
\begin{figure}[tbp]
  \centering
  \input{publication_terminal_v1/figures/tikz/figure_r01_terminal_contrast_forest.tex}
  \caption{...}
  \label{fig:terminal-contrast-forest}
\end{figure}
```

Import a generated table directly:

```tex
\input{publication_terminal_v1/tables/table_r01_final_primary_performance_daily_risk.tex}
```

Use `manuscript_plan/manuscript_artifact_plan.md` as the authoritative page-
budget decision list and `claim_ledger/terminal_claim_ledger.md` when revising
the abstract, results, discussion, and conclusion.
