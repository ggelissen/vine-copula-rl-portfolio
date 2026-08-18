from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from publication_pipeline_draft.generate_terminal_publication_artifacts import (
    FIGURES, generate, parser, resolve_terminal_root)


ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "manuscript_revision_causal_v1/publication_terminal_v1"


def available_terminal_results() -> Path | None:
    candidates = (
        ROOT / "analysis_outputs/terminal_robustness_v1",
        ROOT / (
            "analysis_work/terminal_robustness_v1_review/analysis_outputs/"
            "terminal_robustness_v1"),
    )
    return next((path for path in candidates
                 if (path / "terminal_robustness_manifest.json").is_file()), None)


def test_terminal_publication_generation_is_additive(tmp_path: Path) -> None:
    terminal = available_terminal_results()
    if terminal is None:
        pytest.skip("Immutable terminal robustness results are not present")
    existing = ROOT / "manuscript_revision_causal_v1/figures/tikz/figure_manifest.json"
    before = existing.read_bytes()
    output = tmp_path / "terminal-publication"
    arguments = parser().parse_args([
        "--repo-root", str(ROOT),
        "--terminal-results", str(terminal),
        "--output", str(output),
    ])
    manifest = generate(arguments)
    assert manifest["status"] == "terminal_publication_artifacts_generated"
    assert manifest["additive_only"] is True
    assert manifest["existing_publication_artifacts_modified"] is False
    assert existing.read_bytes() == before
    assert len(list((output / "figures/tikz").glob("figure_r*.tex"))) == len(FIGURES)


def test_generated_terminal_tikz_is_native_and_balanced() -> None:
    if not GENERATED.is_dir():
        pytest.skip("Generated terminal publication bundle is not present")
    figures = sorted((GENERATED / "figures/tikz").glob("figure_r*.tex"))
    assert len(figures) == len(FIGURES)
    for path in figures:
        source = path.read_text(encoding="utf-8")
        assert source.count(r"\begin{tikzpicture}") == 1
        assert source.count(r"\end{tikzpicture}") == 1
        assert r"\includegraphics" not in source


def test_terminal_preamble_supports_hpc_texlive_2019() -> None:
    preamble = GENERATED / "figures/tikz/tikz_preamble.tex"
    if not preamble.is_file():
        pytest.skip("Generated terminal publication bundle is not present")
    source = preamble.read_text(encoding="utf-8")
    assert r"\pgfplotsset{compat=1.16}" in source
    assert r"\pgfplotsset{compat=1.18}" not in source


def test_terminal_figures_reserve_space_for_labels() -> None:
    figure_root = GENERATED / "figures/tikz"
    if not figure_root.is_dir():
        pytest.skip("Generated terminal publication bundle is not present")
    forest = (figure_root / "figure_r01_terminal_contrast_forest.tex").read_text(
        encoding="utf-8")
    risk = (figure_root / "figure_r02_intramonth_risk.tex").read_text(
        encoding="utf-8")
    friction = (figure_root / "figure_r03_friction_surface.tex").read_text(
        encoding="utf-8")
    pretraining = (figure_root / "figure_r04_pretraining_tradeoff.tex").read_text(
        encoding="utf-8")
    stability = (figure_root / "figure_r05_resampling_stability.tex").read_text(
        encoding="utf-8")

    assert r"width=0.72\linewidth" in forest
    assert r"width=0.31\linewidth" in risk
    assert r"width=0.38\linewidth" in risk
    assert "horizontal sep=2.00cm" in risk
    assert r"\addlegendentry" not in risk
    assert "legend style=" not in risk
    assert "axis y line*=right" not in risk
    assert "yticklabel pos=right" not in risk
    assert "title={(a) Drawdown horizon}" in risk
    assert "title={(b) Daily tail risk}" in risk
    assert "yshift=3pt" in risk and "yshift=-3pt" in risk
    assert "Equal weight" in risk and "NN-vine TD3" in risk
    assert friction.count(r"ylabel={Annual CRRA CE (\%)}") == 1
    assert "legend columns=2" in friction
    assert "ytick={20,22,24,26,28,30,32}" not in friction
    assert "ytick={18,20,22,24,26,28,30,32,34}" not in friction
    assert r"width=0.41\linewidth" in pretraining
    assert pretraining.count(r"ylabel={Ensemble annual CRRA CE (\%)}") == 1
    assert "Vine synthetic" in pretraining and "Block bootstrap" in pretraining
    assert "CI above zero" not in stability
    assert "Above zero" in stability and "Crosses zero" in stability
    assert "rotate=45" not in stability
    assert "at (3.5,-8.55)" in stability
    assert "at (7.0,-8.55)" in stability


def test_claim_ledger_enforces_evidence_boundaries() -> None:
    ledger = GENERATED / "claim_ledger/terminal_claim_ledger.csv"
    if not ledger.is_file():
        pytest.skip("Generated terminal publication bundle is not present")
    rows = {row["claim_id"]: row for row in csv.DictReader(
        ledger.open(encoding="utf-8", newline=""))}
    assert len(rows) == 12
    assert rows["TR-C02"]["decision"] == "rejected"
    assert rows["TR-C03"]["decision"] == "opposite_direction_evidence"
    assert rows["TR-C05"]["decision"] == "not_established"
    assert rows["TR-C12"]["decision"] == "prohibited"
    assert "superior" in rows["TR-C02"]["prohibited_wording"].lower()


def test_page_plan_is_strict_and_nonredundant() -> None:
    plan = GENERATED / "manuscript_plan/manuscript_artifact_plan.csv"
    if not plan.is_file():
        pytest.skip("Generated terminal publication bundle is not present")
    rows = list(csv.DictReader(plan.open(encoding="utf-8", newline="")))
    main = [row for row in rows if row["decision"] == "main_text"]
    assert len(main) == 7
    assert any(row["artifact"] == "figure_r01_terminal_contrast_forest.tex"
               for row in main)
    assert any(row["artifact"] == "table_r01_final_primary_performance_daily_risk.tex"
               for row in main)
    assert next(row for row in rows if row["artifact"] ==
                "figure_11_causal_forest.tex")["decision"] == "omit"


def test_publication_manifest_does_not_create_confirmation() -> None:
    path = GENERATED / "publication_artifact_manifest.json"
    if not path.is_file():
        pytest.skip("Generated terminal publication bundle is not present")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["confirmatory_claim_created"] is False
    assert manifest["artifact_count"] >= 25
