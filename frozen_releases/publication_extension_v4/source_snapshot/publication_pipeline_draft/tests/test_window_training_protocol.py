from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from publication_pipeline_draft.window_training_protocol import materialize


ROOT = Path(__file__).resolve().parents[2]
PROGRAM = ROOT / "publication_pipeline_draft/config/publication_research_program_v2.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_window_contract_has_five_algorithms_and_matched_seeds(
        tmp_path: Path) -> None:
    program = json.loads(PROGRAM.read_text(encoding="utf-8"))
    panel = program["panels"][0]
    window = tmp_path / "window"
    window.mkdir()
    returns = window / "window_daily_log_returns.csv"
    returns.write_text("date," + ",".join(panel["asset_order"]) + "\n",
                       encoding="utf-8")
    manifest = {
        "release_status": "frozen_window_return_input_no_confirmation",
        "confirmatory_claim_permitted": False,
        "panel_id": panel["panel_id"], "window_id": "w01",
        "evidence_class": "retrospective_walk_forward",
        "asset_order": panel["asset_order"], "asset_count": 18,
        "reference_asset_index_1based": 18, "vine_truncation_level": 17,
        "return_file_sha256": digest(returns),
    }
    input_manifest = window / "return_input_manifest.json"
    input_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    readme = window / "READ_ONLY_WINDOW_INPUT.txt"
    readme.write_text("development\n", encoding="utf-8")
    (window / "CONTENTS.sha256").write_text(
        "\n".join(f"{digest(path)}  {path.name}"
                  for path in (readme, input_manifest, returns)) + "\n",
        encoding="ascii",
    )
    output = tmp_path / "contract"
    result = materialize(ROOT, PROGRAM, window, Path("data/external"), output)
    with (output / "window_rl_jobs.csv").open(newline="", encoding="utf-8") as stream:
        jobs = list(csv.DictReader(stream))
    assert result["job_count"] == len(jobs) == 50
    assert {row["algorithm"] for row in jobs} == {"td3", "ddpg", "sac", "ppo", "a2c"}
    assert all(row["RETURNS_DATA_KIND"] == "daily_log_returns" for row in jobs)
    assert all(row["REF_COL"] == "18" for row in jobs)
    for algorithm in {row["algorithm"] for row in jobs}:
        assert len({row["seed"] for row in jobs if row["algorithm"] == algorithm}) == 10


def test_scalability_code_has_no_factorial_order_search() -> None:
    source = (ROOT / "benchmark_models/dynamic_vine_NN.r").read_text(
        encoding="utf-8")
    assert 'if (d > 9L)' in source
    assert '"deterministic_all_start_greedy_2opt"' in source
    assert "truncation_level = Inf" in source
    assert '"nn_dynamic_truncated_t_vine"' in source
