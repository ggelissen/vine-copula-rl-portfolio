#!/usr/bin/env python3
"""Build additive paper tables, TikZ figures, narrative, and claim controls.

The generator never writes into the existing TikZ bundle or manuscript chapters.
It transforms the immutable terminal-robustness tables into a separate, atomic
publication bundle so authors can decide what to import.
"""

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

from publication_pipeline_draft.tikz_figures.style import STYLE
from publication_pipeline_draft.terminal_publication import (
    artifact_plan, claims, narrative, tables)
from publication_pipeline_draft.terminal_publication.common import (
    PublicationContext, TerminalPublicationError, require, verify_contents,
    write_manifest)


# The HPC image currently ships TeX Live 2019 / PGFPlots 1.16.  The terminal
# figures use no 1.17+ features, so emit an isolated backward-compatible copy
# of the visual system without altering the existing manuscript preamble.
TERMINAL_STYLE = STYLE.replace(
    r"\pgfplotsset{compat=1.18}",
    r"\pgfplotsset{compat=1.16}",
)
require(TERMINAL_STYLE != STYLE,
        "Could not apply the terminal PGFPlots compatibility override")


FIGURES = (
    "figure_r01_contrast_forest",
    "figure_r02_intramonth_risk",
    "figure_r03_friction_surface",
    "figure_r04_pretraining_tradeoff",
    "figure_r05_resampling_stability",
)


def resolve_terminal_root(repo: Path, requested: Path | None) -> Path:
    if requested is not None:
        candidate = requested if requested.is_absolute() else repo / requested
        return candidate.resolve()
    candidates = (
        repo / "analysis_outputs/terminal_robustness_v1",
        repo / (
            "analysis_work/terminal_robustness_v1_review/analysis_outputs/"
            "terminal_robustness_v1"),
    )
    for candidate in candidates:
        if (candidate / "terminal_robustness_manifest.json").is_file():
            return candidate.resolve()
    raise TerminalPublicationError(
        "Terminal robustness results were not found. Pass --terminal-results "
        "with the directory containing terminal_robustness_manifest.json.")


def validate_tikz(context: PublicationContext) -> None:
    figures = [item for item in context.artifacts
               if item["artifact_type"] == "tikz_figure"]
    require(len(figures) == len(FIGURES),
            f"Expected {len(FIGURES)} terminal figures, found {len(figures)}")
    for item in figures:
        path = context.output / item["file"]
        source = path.read_text(encoding="utf-8")
        require(source.count(r"\begin{tikzpicture}") == 1,
                f"Expected one tikzpicture in {path}")
        require(source.count(r"\end{tikzpicture}") == 1,
                f"Unbalanced tikzpicture in {path}")
        require(r"\includegraphics" not in source,
                f"Raster include found in terminal TikZ figure: {path}")
        require(re.search(r"(?<![A-Za-z])(?:nan|[+-]?inf)(?![A-Za-z])",
                          source.lower()) is None,
                f"Non-finite token found in terminal TikZ figure: {path}")
        for options in re.findall(
                r"\begin\{(?:axis|groupplot)\}\[(.*?)\]",
                source, flags=re.DOTALL):
            require(re.search(r"\n[ \t]*\n", options) is None,
                    f"Blank paragraph inside PGFPlots options in {path}")


def write_support_files(context: PublicationContext) -> None:
    context.write_text(
        "figures/tikz/tikz_preamble.tex", TERMINAL_STYLE,
        artifact_type="tikz_preamble",
        title="Terminal TikZ shared preamble",
        evidence_class="presentation_only")
    figure_items = [item for item in context.artifacts
                    if item["artifact_type"] == "tikz_figure"]
    preview = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[margin=16mm]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\input{tikz_preamble.tex}",
        r"\begin{document}",
        r"\section*{Terminal publication figure proof}",
    ]
    for index, item in enumerate(figure_items):
        if index:
            preview.append(r"\clearpage")
        preview.extend((
            f"\\subsection*{{{item['title']}}}",
            r"\begin{center}",
            f"\\input{{{Path(item['file']).name}}}",
            r"\end{center}",
        ))
    preview.append(r"\end{document}")
    context.write_text(
        "figures/tikz/preview_terminal_figures.tex", "\n".join(preview),
        artifact_type="latex_preview",
        title="Terminal TikZ proof document",
        evidence_class="presentation_only")
    readme = """# Terminal publication artifact bundle

This directory is additive. It does not overwrite the existing manuscript or
the existing `manuscript_revision_causal_v1/figures/tikz` bundle.

- `tables/` contains publication CSV and LaTeX tables.
- `figures/tikz/` contains five new terminal-evidence figures and a proof file.
- `claim_ledger/` separates permissible findings from prohibited claims.
- `narrative/` contains an additive Results-section fragment.
- `manuscript_plan/` gives strict main-text, appendix, supplement, and omission decisions.

The existing manuscript preamble already defines the same TikZ visual system.
Copy or reference the new figure snippets without inputting a second preamble.
All statistical values are presentation transforms of the immutable terminal
tables; no inference or policy selection is recomputed here.
"""
    context.write_text(
        "README.md", readme, artifact_type="runbook",
        title="Terminal publication bundle instructions",
        evidence_class="authorial_synthesis")


def generate(arguments: argparse.Namespace) -> dict[str, object]:
    repo = arguments.repo_root.resolve()
    require(repo.is_dir(), f"Repository root does not exist: {repo}")
    terminal_root = resolve_terminal_root(repo, arguments.terminal_results)
    verify_contents(terminal_root)
    terminal_manifest_path = terminal_root / "terminal_robustness_manifest.json"
    terminal_manifest = json.loads(terminal_manifest_path.read_text(encoding="utf-8"))
    require(terminal_manifest.get("status") == "terminal_robustness_campaign_complete",
            "Terminal robustness campaign is not complete")
    require(terminal_manifest.get("policy_retraining_performed") is False,
            "Terminal campaign unexpectedly retrained policies")
    require(terminal_manifest.get("model_selection_performed") is False,
            "Terminal campaign unexpectedly performed model selection")
    require(terminal_manifest.get("confirmatory_claim_created") is False,
            "Terminal campaign unexpectedly created a confirmatory claim")

    output = arguments.output
    output = output if output.is_absolute() else repo / output
    output = output.resolve()
    require(output != repo and output != Path(output.anchor),
            "Refusing unsafe publication output path")
    if output.exists() and not arguments.replace:
        raise TerminalPublicationError(
            f"Additive output already exists: {output}. Pass --replace only to "
            "regenerate this terminal bundle; existing publication work is untouched.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    context = PublicationContext(repo=repo, terminal_root=terminal_root,
                                 output=temporary)
    try:
        tables.generate(context)
        claims.generate(context)
        narrative.generate(context)
        artifact_plan.generate(context)
        for module_name in FIGURES:
            module = importlib.import_module(
                f"publication_pipeline_draft.terminal_publication.{module_name}")
            module.generate(context)
        write_support_files(context)
        validate_tikz(context)
        manifest_path = write_manifest(context, terminal_manifest)
        if output.exists():
            shutil.rmtree(output)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return json.loads((output / manifest_path.name).read_text(encoding="utf-8"))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, default=Path("."))
    result.add_argument("--terminal-results", type=Path)
    result.add_argument(
        "--output", type=Path,
        default=Path("manuscript_revision_causal_v1/publication_terminal_v1"))
    result.add_argument("--replace", action="store_true")
    return result


def main() -> int:
    arguments = parser().parse_args()
    try:
        manifest = generate(arguments)
    except TerminalPublicationError as error:
        print(f"TERMINAL PUBLICATION FAILURE: {error}")
        return 1
    print(json.dumps({
        "status": manifest["status"],
        "artifact_count": manifest["artifact_count"],
        "additive_only": manifest["additive_only"],
        "output": str(arguments.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
