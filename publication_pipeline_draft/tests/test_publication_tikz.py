from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from publication_pipeline_draft.generate_publication_tikz import generate, parser
from publication_pipeline_draft.tikz_figures.common import FigureDataError


ROOT = Path(__file__).resolve().parents[2]


def test_publication_tikz_generates_core_bundle(tmp_path: Path) -> None:
    arguments = parser().parse_args([
        "--repo-root", str(ROOT),
        "--output", str(tmp_path / "tikz"),
    ])
    manifest = generate(arguments)
    assert manifest["status"] == "publication_tikz_generated"
    assert manifest["figure_count"] >= 16
    output = tmp_path / "tikz"
    assert (output / "tikz_preamble.tex").is_file()
    assert (output / "preview_all_figures.tex").is_file()
    assert (output / "CONTENTS.sha256").is_file()
    for item in manifest["generated"]:
        source = (output / item["file"]).read_text(encoding="utf-8")
        assert source.count(r"\begin{tikzpicture}") == 1
        assert source.count(r"\end{tikzpicture}") == 1
        for options in re.findall(
                r"\\begin\{(?:axis|groupplot)\}\[(.*?)\]",
                source, flags=re.DOTALL):
            assert re.search(r"\n[ \t]*\n", options) is None


def test_existing_output_requires_replace(tmp_path: Path) -> None:
    output = tmp_path / "tikz"
    output.mkdir()
    arguments = parser().parse_args([
        "--repo-root", str(ROOT),
        "--output", str(output),
    ])
    with pytest.raises(Exception, match="Pass --replace"):
        generate(arguments)


def test_explicit_training_diagnostics_fail_closed(tmp_path: Path) -> None:
    arguments = parser().parse_args([
        "--repo-root", str(ROOT),
        "--output", str(tmp_path / "tikz"),
        "--training-diagnostics", str(tmp_path / "missing-training-input"),
    ])
    with pytest.raises(FigureDataError,
                       match="Explicit training diagnostics could not produce T01"):
        generate(arguments)


def test_generated_manifest_labels_evidence_classes() -> None:
    manifest_path = ROOT / (
        "manuscript_revision_causal_v1/figures/tikz/figure_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert all(item["evidence_class"] for item in manifest["generated"])
    assert any(item["evidence_class"] == "frozen_confirmatory_primary_evaluation"
               for item in manifest["generated"])
    assert any(item["evidence_class"] == "post_holdout_explanatory"
               for item in manifest["generated"])


def test_seed_diagnostics_are_distributional_and_multirun() -> None:
    output = ROOT / "manuscript_revision_causal_v1/figures/tikz"
    robustness = (output / "figure_05_seed_robustness.tex").read_text(
        encoding="utf-8")
    pretraining = (output / "figure_t01_pretraining_stability.tex").read_text(
        encoding="utf-8")
    optimizer = (output / "figure_t02_optimizer_diagnostics.tex").read_text(
        encoding="utf-8")
    assert "rectangle" in robustness
    assert robustness.count("opacity=0.62") == 4
    assert "group size=4 by 1" in robustness
    assert pretraining.count("opacity=0.20") == 80
    assert optimizer.count("opacity=0.18") == 80
    assert pretraining.count("ultra thick") == 4
    assert optimizer.count("ultra thick") == 4


def test_dense_figures_use_compact_nonoverlapping_interfaces() -> None:
    output = ROOT / "manuscript_revision_causal_v1/figures/tikz"
    risk_return = (output / "figure_02_risk_return_utility.tex").read_text(
        encoding="utf-8")
    implementation = (output / "figure_04_implementation.tex").read_text(
        encoding="utf-8")
    assert "legend columns=1" in risk_return
    assert "at={(1.04,0.50)}" in risk_return
    assert "CE " in risk_return
    assert "group size=2 by 1" in implementation
    assert implementation.count(r"\begin{axis}") == 0
    assert r"\begin{groupplot}" in implementation


def test_methodology_diagrams_are_generated_as_tikz() -> None:
    output = ROOT / "manuscript_revision_causal_v1/figures/tikz"
    architecture = (output / "figure_m01_lstm_td3_architecture.tex").read_text(
        encoding="utf-8")
    training = (output / "figure_m02_training_strategy.tex").read_text(
        encoding="utf-8")
    methodology = (ROOT / "manuscript_revision_causal_v1/03_Methodology.tex").read_text(
        encoding="utf-8")
    assert "Step 1: Deterministic Actor" in architecture
    assert "Step 2: Independent Twin Critics" in architecture
    assert r"Projection: $\sum_iw_i=1$, $\|w\|_1\leq1.5$" in architecture
    assert "Step 1: Training-Only Synthetic Bundle" in training
    assert "37 fit trajectories; 23 overlapping paths purged" in training
    assert "all 61 permissible trajectories" in training
    assert "figure_m01_lstm_td3_architecture.tex" in methodology
    assert "figure_m02_training_strategy.tex" in methodology
