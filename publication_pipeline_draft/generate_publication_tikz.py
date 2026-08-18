#!/usr/bin/env python3
"""Generate self-contained, publication-grade TikZ/PGFPlots figure snippets."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from publication_pipeline_draft.tikz_figures.common import (
    FigureContext, FigureDataError, require, write_manifest)
from publication_pipeline_draft.tikz_figures import style


MODULES = (
    "figure_m01_lstm_td3_architecture",
    "figure_m02_training_strategy",
    "figure_01_wealth_drawdown",
    "figure_02_risk_return_utility",
    "figure_03_allocation_heatmap",
    "figure_04_implementation",
    "figure_05_seed_robustness",
    "figure_06_primary_inference",
    "figure_07_monthly_excess",
    "figure_08_ensemble_cancellation",
    "figure_09_ensemble_size",
    "figure_10_drift_accounting",
    "figure_11_causal_forest",
    "figure_12_causal_turnover_performance",
    "figure_13_causal_seed_effects",
    "figure_14_causal_wealth",
    "figure_15_compressed_benchmark_reconciliation",
    "figure_16_focused_walk_forward",
    "figure_s01_marginal_fidelity",
    "figure_s02_dependence_fidelity",
    "figure_s03_temporal_fidelity",
    "figure_s04_seed_correlation",
    "figure_t01_pretraining_stability",
    "figure_t02_optimizer_diagnostics",
)


def optional_path(value: str | None, repo: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def static_validate(context: FigureContext) -> None:
    banned = ("Axes.boxplot", "\\includegraphics")
    for item in context.generated:
        path = context.output / item["file"]
        text = path.read_text(encoding="utf-8")
        require(text.count("\\begin{tikzpicture}") == 1,
                f"Expected one tikzpicture in {path}.")
        require(text.count("\\end{tikzpicture}") == 1,
                f"Unbalanced tikzpicture in {path}.")
        for options in re.findall(
                r"\\begin\{(?:axis|groupplot)\}\[(.*?)\]",
                text, flags=re.DOTALL):
            require(re.search(r"\n[ \t]*\n", options) is None,
                    f"Blank paragraph inside PGFPlots options in {path}.")
        lower = text.lower()
        for token in banned:
            require(token.lower() not in lower,
                    f"Banned or non-finite token {token!r} in {path}.")
        require(re.search(r"(?<![A-Za-z])(?:nan|[+-]?inf)(?![A-Za-z])",
                          lower) is None,
                f"Non-finite numeric token in {path}.")
        require(re.search(r"(?<!\\)\\[0-9]", text) is None,
                f"Suspicious single-backslash numeric command in {path}.")


def write_support_files(context: FigureContext) -> None:
    generated = sorted(context.generated, key=lambda item: item["file"])
    catalog = [
        "# Publication TikZ figure bundle",
        "",
        "These snippets contain inline frozen data and use one shared visual system.",
        "They do not recompute statistics. Each file header declares its evidence class.",
        "",
        "Insert a figure with:",
        "",
        "```tex",
        "\\begin{figure}[tbp]",
        "  \\centering",
        "  \\input{figures/tikz/figure_01_wealth_drawdown.tex}",
        "  \\caption{...}",
        "  \\label{fig:wealth-drawdown}",
        "\\end{figure}",
        "```",
        "",
        "The manuscript preamble must input `figures/tikz/tikz_preamble.tex` once.",
        "Compile `preview_all_figures.tex` from this directory for a visual QA booklet.",
        "",
        "## Generated figures",
        "",
    ]
    for item in generated:
        catalog.append(
            f"- `{item['file']}` — {item['title']} "
            f"(`{item['evidence_class']}`)")
    if context.skipped:
        catalog.extend(("", "## Intentionally skipped", ""))
        catalog.extend(f"- `{item['figure']}` — {item['reason']}"
                       for item in context.skipped)
    (context.output / "README.md").write_text("\n".join(catalog) + "\n",
                                               encoding="utf-8")

    preview = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[margin=16mm]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\input{tikz_preamble.tex}",
        r"\pagestyle{plain}",
        r"\begin{document}",
        r"\section*{Publication figure proof}",
        r"Evidence labels below are part of the proof only, not the manuscript figures.",
    ]
    for index, item in enumerate(generated):
        if index:
            preview.append(r"\clearpage")
        preview.extend((
            f"\\subsection*{{{item['title']}}}",
            f"\\noindent\\textit{{Evidence class: {item['evidence_class'].replace('_', ' ')}}}\\par\\medskip",
            r"\begin{center}",
            f"\\input{{{item['file']}}}",
            r"\end{center}",
        ))
    preview.append(r"\end{document}")
    (context.output / "preview_all_figures.tex").write_text(
        "\n".join(preview) + "\n", encoding="utf-8")


def generate(args: argparse.Namespace) -> dict[str, object]:
    repo = args.repo_root.resolve()
    output = args.output.resolve()
    require(repo.is_dir(), f"Repository root does not exist: {repo}")
    require(output != repo, "Refusing to use the repository root as figure output.")
    require(output != Path(output.anchor),
            "Refusing to use a filesystem root as figure output.")
    if output.exists() and not args.replace:
        raise FigureDataError(
            f"Output exists: {output}. Pass --replace to regenerate it atomically.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.",
                                      dir=output.parent))
    context = FigureContext(
        repo=repo,
        output=temporary,
        main_path=optional_path(args.main_results, repo),
        ensemble_path=optional_path(args.ensemble_results, repo),
        causal_path=optional_path(args.causal_archive, repo),
        reconciliation_path=optional_path(args.reconciliation_results, repo),
        synthetic_path=optional_path(args.synthetic_diagnostics, repo),
        training_path=optional_path(args.training_diagnostics, repo),
        focused_path=optional_path(args.focused_results, repo),
    )
    try:
        style.generate(context)
        for module_name in MODULES:
            module = importlib.import_module(
                f"publication_pipeline_draft.tikz_figures.{module_name}")
            module.generate(context)
        require(len(context.generated) >= 16,
                "Core publication evidence did not produce all required figures.")
        static_validate(context)
        write_support_files(context)
        write_manifest(context)
        if output.exists():
            shutil.rmtree(output)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    manifest = json.loads((output / "figure_manifest.json").read_text(
        encoding="utf-8"))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, default=Path("."))
    result.add_argument(
        "--output", type=Path,
        default=Path("manuscript_revision_causal_v1/figures/tikz"))
    result.add_argument("--replace", action="store_true")
    result.add_argument("--main-results")
    result.add_argument("--ensemble-results")
    result.add_argument("--causal-archive")
    result.add_argument("--reconciliation-results")
    result.add_argument("--synthetic-diagnostics")
    result.add_argument("--training-diagnostics")
    result.add_argument("--focused-results")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        manifest = generate(args)
    except FigureDataError as error:
        print(f"PUBLICATION TIKZ FAILURE: {error}")
        return 1
    print(json.dumps({
        "status": manifest["status"],
        "figure_count": manifest["figure_count"],
        "skipped_count": manifest["skipped_count"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
