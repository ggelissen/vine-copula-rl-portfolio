from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from publication_pipeline_draft.publication_pipeline import sha256_file


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "frozen_releases/mixed_pretraining_response_v1_evidence_v1"
GENERATED = ROOT / (
    "manuscript_revision_causal_v1/publication_mixed_pretraining_v1")


def verify_inventory(root: Path) -> None:
    for line in (root / "CONTENTS.sha256").read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        path = root / relative.strip().lstrip("*")
        assert path.is_file()
        assert sha256_file(path) == expected


def test_frozen_mixed_pretraining_release_is_fail_closed() -> None:
    if not RELEASE.is_dir():
        pytest.skip("Frozen mixed-pretraining evidence is not present")
    verify_inventory(RELEASE)
    manifest = json.loads((
        RELEASE / "mixed_pretraining_evidence_manifest.json"
    ).read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen_mixed_pretraining_evidence_v1"
    assert manifest["evidence_class"] == "post_holdout_explanatory"
    assert manifest["confirmatory_claim_permitted"] is False
    assert manifest["same_holdout_further_tuning_authorized"] is False
    assert manifest["checkpoint_inventory"]["full_checkpoint_count"] == 10
    assert manifest["checkpoint_inventory"]["pretrained_checkpoint_count"] == 10


def test_publication_extension_is_additive_and_training_free() -> None:
    if not GENERATED.is_dir():
        pytest.skip("Mixed-pretraining publication extension is not present")
    verify_inventory(GENERATED)
    manifest = json.loads((
        GENERATED / "mixed_pretraining_publication_manifest.json"
    ).read_text(encoding="utf-8"))
    assert manifest["status"] == "mixed_pretraining_publication_artifacts_generated"
    assert manifest["additive_only"] is True
    assert manifest["existing_publication_artifacts_modified"] is False
    assert manifest["model_training_performed"] is False
    assert manifest["model_selection_performed"] is False
    assert manifest["confirmatory_claim_created"] is False


def test_mixed_claim_ledger_preserves_evidence_boundaries() -> None:
    path = GENERATED / "claim_ledger/mixed_pretraining_claim_ledger.csv"
    if not path.is_file():
        pytest.skip("Mixed-pretraining claim ledger is not present")
    rows = {row["claim_id"]: row for row in csv.DictReader(
        path.open(encoding="utf-8", newline=""))}
    assert len(rows) == 7
    assert rows["MP-C02"]["decision"] == "not_established"
    assert rows["MP-C05"]["decision"] == "rejected"
    assert rows["MP-C06"]["decision"] == "prohibited"
    assert rows["MP-C07"]["decision"] == "supported_descriptively"
    assert "superior" in rows["MP-C01"]["prohibited_wording"].lower()


def test_leave_one_seed_out_is_complete_and_cost_approximation_is_refused() -> None:
    loo = GENERATED / "robustness/mixed_leave_one_seed_out.csv"
    if not loo.is_file():
        pytest.skip("Frozen-weight robustness outputs are not present")
    loo_rows = list(csv.DictReader(loo.open(encoding="utf-8", newline="")))
    assert len(loo_rows) == 10
    assert len({int(row["omitted_seed"]) for row in loo_rows}) == 10
    assert all(int(row["retained_seeds"]) == 9 for row in loo_rows)
    assert not (GENERATED / "robustness/mixed_transaction_cost_grid.csv").exists()
    manifest = json.loads((
        GENERATED / "mixed_pretraining_publication_manifest.json"
    ).read_text(encoding="utf-8"))
    assert manifest["transaction_cost_rescoring_status"].startswith("not_generated")


def test_mixed_figure_is_native_tikz_and_shows_seed_dispersion() -> None:
    path = GENERATED / "figures/tikz/figure_mp01_mixed_pretraining_evidence.tex"
    if not path.is_file():
        pytest.skip("Mixed-pretraining TikZ figure is not present")
    source = path.read_text(encoding="utf-8")
    assert source.count(r"\begin{tikzpicture}") == 1
    assert source.count(r"\end{tikzpicture}") == 1
    assert r"\includegraphics" not in source
    assert "Matched-seed effects" in source
    assert "MBB(3) 95\\% CI" in source
    assert "horizontal sep=2.35cm" in source
    assert r"width=0.40\linewidth" in source
    assert r"width=0.44\linewidth" in source
    assert "rotate=18" not in source
    assert "align=center" in source
    assert "font=\\tiny" in source
    assert "column sep=20pt" in source
    assert "black,dashed,forget plot" in source


def test_hpc_workflow_exposes_freeze_and_publication_steps() -> None:
    source = (ROOT / "hpc/run_mixed_pretraining_response_v1.sh").read_text(
        encoding="utf-8")
    assert "freeze-evidence" in source
    assert "publication)" in source
    assert "freeze_mixed_pretraining_evidence" in source
    assert "generate_mixed_pretraining_publication_artifacts" in source
