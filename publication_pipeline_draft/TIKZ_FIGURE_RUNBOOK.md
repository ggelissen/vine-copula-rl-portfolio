# Publication TikZ figure workflow

The figure pipeline converts already-computed CSV evidence into vector-native
TikZ/PGFPlots snippets. It does not refit a model, rescore a return, or alter a
statistical result. Every generated file carries an evidence-class header and
the manifest records SHA-256 hashes of the source inputs.

The visual system is original project code informed by the curated
[awesome-tikz](https://github.com/xiaohanyu/awesome-tikz) resource index and the
official [PGFPlots manual](https://tikz.dev/pgfplots/). It uses grouped panels,
date axes, diverging matrix plots, confidence bands, interval forests, direct
labels, and evidence-class annotations. No gallery template was copied. Color
is redundant with marks and line patterns, grid ink is deliberately subdued,
and headline series receive emphasis without suppressing comparator data.

## Generate all currently available figures

Run this from the repository root:

```bash
python3 -m publication_pipeline_draft.generate_publication_tikz \
  --repo-root . \
  --output manuscript_revision_causal_v1/figures/tikz \
  --replace
```

The default inputs point to the preserved main OOS, ensemble-mechanism, causal,
and reconciliation outputs in this repository. Optional diagnostic locations
can be supplied without copying data into the repository:

```bash
python3 -m publication_pipeline_draft.generate_publication_tikz \
  --repo-root . \
  --output manuscript_revision_causal_v1/figures/tikz \
  --synthetic-diagnostics /path/to/synthetic_diagnostics \
  --training-diagnostics /path/to/publication_training_artifacts \
  --focused-results analysis_outputs/focused_walk_forward_mechanisms_v1 \
  --replace
```

The synthetic directory must contain `fidelity_metrics.csv`,
`correlation_comparison.csv`, `tail_dependence_comparison.csv`, and
`temporal_dependence.csv`. The training directory must contain either `raw/`
aggregates from `diagnostic_artifacts.py` or the two aggregate CSVs at its root.
Unavailable optional evidence is recorded as skipped; no placeholder data are
ever generated.

## Coverage and exclusions

| Earlier output family | TikZ replacement |
|---|---|
| Main OOS wealth and drawdown | `figure_01_wealth_drawdown.tex` |
| Risk-return, allocations, implementation, seeds, inference, monthly excess | `figure_02` through `figure_07` |
| Ensemble cancellation, size sensitivity, and drift-aware accounting | `figure_08` through `figure_10` |
| Pairwise seed-weight correlation | `figure_s04_seed_correlation.tex` |
| Causal effect, economics, seed heterogeneity, and wealth | `figure_11` through `figure_14` |
| Post-hoc benchmark reconciliation | `figure_15_compressed_benchmark_reconciliation.tex` |
| Focused walk-forward mechanism result | `figure_16_focused_walk_forward.tex` once its results exist |
| Synthetic fidelity diagnostics | `figure_s01` through `figure_s03` |
| Training and optimizer diagnostics | `figure_t01` and `figure_t02` |

The low-level RL wealth and weight PDFs are superseded by the common-accounting
wealth/drawdown and allocation figures. Legacy `eval/ablation.r` and
`eval/sensitivity.r` charts are intentionally excluded: they are not empirical
evidence from the frozen causal experiment and must not enter the paper. The
pipeline retains the earlier PDF/PNG writers for backward-compatible diagnostics,
but the manuscript-facing visual layer is exclusively the generated TikZ bundle.

## Manuscript use

`manuscript_revision_causal_v1/main.tex` imports the generated visual preamble.
Insert any snippet with a normal LaTeX figure environment:

```tex
\begin{figure}[tbp]
  \centering
  \input{figures/tikz/figure_01_wealth_drawdown.tex}
  \caption{Common-path net wealth and drawdown in the frozen evaluation.}
  \label{fig:oos-wealth-drawdown}
\end{figure}
```

Do not resize a TikZ figure with `\resizebox`; its fonts and line weights are
already coordinated with the manuscript. If a journal requires grayscale, the
line styles and plot marks preserve identification without color.

## Visual QA

Compile `manuscript_revision_causal_v1/figures/tikz/preview_all_figures.tex`
from its own directory. With a full TeX Live installation:

```bash
cd manuscript_revision_causal_v1/figures/tikz
latexmk -pdf -interaction=nonstopmode -halt-on-error preview_all_figures.tex
```

Then inspect every page at normal print scale for label collisions and journal
column fit. The local development machine used for this revision did not expose
a TeX engine, so the generated sources were statically validated but still
require this real compile-and-render check.

## Reproducibility checks

```bash
cd manuscript_revision_causal_v1/figures/tikz
sha256sum -c CONTENTS.sha256
python3 -m json.tool figure_manifest.json
```

`CONTENTS.sha256` protects the complete generated bundle. Regeneration is
atomic: a failed run leaves the previous output directory intact.
