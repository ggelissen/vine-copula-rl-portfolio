from __future__ import annotations

import json
from pathlib import Path
import sys
import csv

import pytest
import yaml

from publication_pipeline_draft.freeze_evaluation_release import (
    BENCHMARK_IDS,
    evaluation_hash_aggregates,
    parse_args,
)
from publication_pipeline_draft.freeze_training_release import sha256_file
from publication_pipeline_draft.locked_evaluation_batch import (
    seed_checkpoints,
    verify_frozen_sources,
)
from publication_pipeline_draft.publication_pipeline import ProtocolError


ROOT = Path(__file__).resolve().parents[2]
NO_VINE_SPECIFICATION_PATH = ROOT / "config/no_vine_ablation_seeds.yaml"


def no_vine_ablation_enabled() -> bool:
    """Only enforce secondary-ablation checks when its protocol is enabled."""
    if not NO_VINE_SPECIFICATION_PATH.is_file():
        return False
    specification = yaml.safe_load(NO_VINE_SPECIFICATION_PATH.read_text()) or {}
    return specification.get("vine_observation_mode") == "zero"


requires_no_vine_ablation = pytest.mark.skipif(
    not no_vine_ablation_enabled(),
    reason=(
        "Secondary no-vine ablation is not enabled in this checkout; "
        "it is excluded from the main evaluation release."
    ),
)


def test_benchmark_and_evaluation_contracts_share_mandate() -> None:
    evaluation = json.loads(
        (ROOT / "publication_pipeline_draft/config/evaluation_contract.json").read_text()
    )
    benchmark = json.loads(
        (ROOT / "publication_pipeline_draft/config/benchmark_contract.json").read_text()
    )
    shared = [
        "evaluation_id", "net_exposure", "gross_leverage", "max_long_weight",
        "max_short_weight", "weight_tolerance", "turnover_cost",
        "annual_short_borrow_rate", "annual_cash_borrow_rate", "crra_gamma",
    ]
    assert {name: evaluation[name] for name in shared} == {
        name: benchmark[name] for name in shared
    }
    assert tuple(BENCHMARK_IDS) == (
        "equal_weight", "shrinkage_mean_variance", "dcc_garch",
        "static_vine", "rolling_vine", "dynamic_nn_vine",
    )
    ensembles = evaluation["predeclared_ensembles"]
    assert [item["strategy_id"] for item in ensembles] == ["vine_td3_ensemble"]
    assert ensembles[0]["minimum_members"] == 20
    assert benchmark["optimizer_allowed_convergence_codes"] == [1, 2, 3, 4]


def test_main_evaluation_freeze_does_not_require_no_vine_release(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_evaluation_release.py",
            "--repo-root", ".",
            "--full-training-release", "frozen_releases/training_schema5_v1",
            "--output", "frozen_releases/evaluation_main_v1",
        ],
    )
    arguments = parse_args()
    assert arguments.no_vine_training_release is None


def test_legacy_full_mode_is_accepted_but_legacy_zero_mode_is_not(tmp_path) -> None:
    seed_directory = tmp_path / "seeds" / "seed_101"
    seed_directory.mkdir(parents=True)
    (seed_directory / "td3_lstm_vine_full.pt").write_bytes(b"checkpoint")
    rows = seed_checkpoints(tmp_path, 1, "full")
    assert rows[0]["mode"] == "full_legacy"
    with pytest.raises(ProtocolError):
        seed_checkpoints(tmp_path, 1, "zero")


@requires_no_vine_ablation
def test_no_vine_ablation_has_ten_distinct_preregistered_seeds() -> None:
    specification = yaml.safe_load(
        NO_VINE_SPECIFICATION_PATH.read_text()
    )
    assert specification["vine_observation_mode"] == "zero"
    seeds = specification["seeds"]
    assert len(seeds) == len(set(seeds)) == 10
    assert specification["minimum_successful_seeds"] == 10
    assert not set(seeds).intersection(range(20260741, 20260761))


@requires_no_vine_ablation
def test_no_vine_ablation_masks_all_policy_visible_vine_signals() -> None:
    environment = (ROOT / "rl/rl_environment.r").read_text()
    sanity = (ROOT / "rl/training_sanity_check.r").read_text()
    trainer = (ROOT / "rl/train_rl.r").read_text()
    evaluator = (ROOT / "rl/evaluate_rl.r").read_text()
    policy_server = (ROOT / "rl/policy_inference_server.py").read_text()

    masking_rule = "cvar_observation <- if (no_vine_observation) 0"
    assert masking_rule in environment
    assert masking_rule in sanity
    for source in (trainer, sanity, policy_server):
        assert "explicit_vine_and_scenario_cvar_v1" in source
    assert 'Sys.getenv("VINE_OBSERVATION_MODE", "full")' in evaluator


def test_locked_batch_is_part_of_the_frozen_source_set() -> None:
    from publication_pipeline_draft.freeze_evaluation_release import EVALUATION_SOURCES

    required = {
        "publication_pipeline_draft/benchmark_weights.R",
        "publication_pipeline_draft/generate_benchmark_weights.R",
        "publication_pipeline_draft/locked_evaluation_batch.py",
        "rl/evaluate_rl.r",
        "rl/action_projection.py",
        "rl/policy_inference_server.py",
    }
    assert required.issubset(set(EVALUATION_SOURCES))
    assert all((ROOT / relative).is_file() for relative in EVALUATION_SOURCES)


def _write_evaluation_release(
    tmp_path: Path, *, hash_schema: int
) -> tuple[Path, Path, dict[str, str]]:
    repo = tmp_path / f"repo_{hash_schema}"
    release = tmp_path / f"release_{hash_schema}"
    rows = []
    sources = [
        ("engine.py", "code", b"print('engine')\n"),
        ("config/settings.json", "config", b'{"setting": 1}\n'),
        ("RUNBOOK.md", "supporting_source", b"frozen protocol\n"),
    ]
    for relative, role, content in sources:
        live = repo / relative
        frozen = release / "source_snapshot" / relative
        live.parent.mkdir(parents=True, exist_ok=True)
        frozen.parent.mkdir(parents=True, exist_ok=True)
        live.write_bytes(content)
        frozen.write_bytes(content)
        rows.append(
            {
                "path": relative,
                "role": role,
                "size_bytes": len(content),
                "sha256": sha256_file(frozen),
            }
        )
    with (release / "evaluation_source_inventory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["path", "role", "size_bytes", "sha256"]
        )
        writer.writeheader()
        writer.writerows(rows)
    aggregates = evaluation_hash_aggregates(rows)
    manifest = {
        "schema_version": 1,
        "release_status": "frozen_pre_holdout_evaluation",
        "holdout_accessed_by_freezer": False,
        "evaluation_source_count": len(rows),
        "evaluation_code_contract_sha256": aggregates[
            "evaluation_source_aggregate_sha256"
        ],
    }
    if hash_schema == 2:
        manifest.update(
            {
                "evaluation_hash_schema_version": 2,
                **aggregates,
                "evaluation_contents_sha256_sidecar": "CONTENTS.sha256.sha256",
            }
        )
    (release / "evaluation_release_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    checksum_rows = [
        f"{sha256_file(path)}  {path.relative_to(release).as_posix()}"
        for path in sorted(item for item in release.rglob("*") if item.is_file())
        if path.name != "CONTENTS.sha256"
    ]
    contents = release / "CONTENTS.sha256"
    contents.write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")
    if hash_schema == 2:
        (release / "CONTENTS.sha256.sha256").write_text(
            f"{sha256_file(contents)}  CONTENTS.sha256\n", encoding="utf-8"
        )
    return repo, release, aggregates


def test_new_evaluation_release_separates_all_hash_roles(tmp_path: Path) -> None:
    repo, release, expected = _write_evaluation_release(tmp_path, hash_schema=2)
    verified = verify_frozen_sources(repo, release)
    assert verified["evaluation_hash_schema_version"] == 2
    assert verified["evaluation_code_sha256"] == expected["evaluation_code_sha256"]
    assert verified["evaluation_config_sha256"] == expected["evaluation_config_sha256"]
    assert verified["evaluation_source_aggregate_sha256"] == expected[
        "evaluation_source_aggregate_sha256"
    ]
    assert verified["evaluation_contents_sha256"] == sha256_file(
        release / "CONTENTS.sha256"
    )
    assert len(
        {
            verified["evaluation_code_sha256"],
            verified["evaluation_config_sha256"],
            verified["evaluation_source_aggregate_sha256"],
            verified["evaluation_contents_sha256"],
        }
    ) == 4


def test_locked_batch_accepts_legacy_v4_aggregate_role(tmp_path: Path) -> None:
    repo, release, expected = _write_evaluation_release(tmp_path, hash_schema=1)
    verified = verify_frozen_sources(repo, release)
    legacy = expected["evaluation_source_aggregate_sha256"]
    assert verified["evaluation_hash_schema_version"] == 1
    assert verified["evaluation_source_aggregate_sha256"] == legacy
    assert verified["evaluation_code_sha256"] == legacy
    assert verified["evaluation_config_sha256"] == legacy
    assert verified["evaluation_contents_sha256"] == sha256_file(
        release / "CONTENTS.sha256"
    )


def test_locked_batch_passes_each_frozen_seed_directory_explicitly() -> None:
    launcher = (ROOT / "evaluate_with_config.r").read_text()
    batch = (ROOT / "publication_pipeline_draft/locked_evaluation_batch.py").read_text()
    assert "model_dir_override" in launcher
    assert "Explicit evaluation model directory:" in launcher
    assert '"config/config.yaml", str(item["directory"])' in batch


def test_historical_evaluator_isolates_r_and_python_libtorch() -> None:
    evaluator = (ROOT / "rl/evaluate_rl.r").read_text()
    server = (ROOT / "rl/policy_inference_server.py").read_text()
    assert "policy_inference_server.py" in evaluator
    assert "run_isolated_policy" in evaluator
    assert "py_run_string(" not in evaluator
    assert "library(reticulate)" not in evaluator
    assert "file_ipc_isolated_libtorch_v1" in server
    assert "supports_vine_observation_mode" in evaluator
    assert "EVAL_DEVELOPMENT_DRY_RUN" in evaluator
    assert "tail(period_split$train, T_eval)" in evaluator
    assert "all_logs <- as.data.frame(rbindlist(" in evaluator
