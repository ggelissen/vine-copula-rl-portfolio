from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import math
import platform
from pathlib import Path

import pytest
import publication_pipeline_draft.future_confirmatory_protocol as protocol_module

from publication_pipeline_draft.future_confirmatory_protocol import (
    ProtocolError,
    execute_contract,
    sha256_file,
    validate_contract_schema,
    validate_evaluation_panel,
    validate_full_contract,
    validate_solver_audit,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "publication_pipeline_draft/config/future_confirmatory_contract.example.json"


RUNNER_SOURCE = r'''#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--test-data", required=True)
parser.add_argument("--model-release", required=True)
parser.add_argument("--hyperparameters", required=True)
parser.add_argument("--selection-manifest", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

with open(args.test_data, newline="", encoding="utf-8") as stream:
    source = list(csv.DictReader(stream))
assets = [name.removeprefix("g_") for name in source[0] if name.startswith("g_")]
output = Path(args.output)
output.mkdir(parents=True, exist_ok=False)
panel_fields = [
    "strategy_id", "window_id", "decision_date", "holding_end_date",
    "calendar_days", "gross_return", "net_return", "turnover",
    "transaction_cost", "financing_cost", "short_notional",
    "cash_borrow_notional", "gross_exposure", "net_exposure",
    *[f"target_w_{asset}" for asset in assets],
    *[f"pretrade_w_{asset}" for asset in assets],
    *[f"g_{asset}" for asset in assets],
]
rows = []
pretrade = [1.0 / len(assets)] * len(assets)
for source_row in source:
    target = [1.0 / len(assets)] * len(assets)
    gross = [float(source_row[f"g_{asset}"]) for asset in assets]
    turnover = sum(abs(left - right) for left, right in zip(target, pretrade))
    gross_return = sum(weight * value for weight, value in zip(target, gross)) - 1.0
    decision = dt.date.fromisoformat(source_row["decision_date"])
    holding = dt.date.fromisoformat(source_row["holding_end_date"])
    row = {
        "strategy_id": "equal_weight", "window_id": source_row["window_id"],
        "decision_date": source_row["decision_date"],
        "holding_end_date": source_row["holding_end_date"],
        "calendar_days": (holding - decision).days,
        "gross_return": gross_return, "net_return": gross_return,
        "turnover": turnover, "transaction_cost": 0.001 * turnover,
        "financing_cost": 0.0, "short_notional": 0.0,
        "cash_borrow_notional": 0.0, "gross_exposure": 1.0,
        "net_exposure": 1.0,
    }
    for asset, value in zip(assets, target):
        row[f"target_w_{asset}"] = value
    for asset, value in zip(assets, pretrade):
        row[f"pretrade_w_{asset}"] = value
    for asset, value in zip(assets, gross):
        row[f"g_{asset}"] = value
    rows.append(row)
    denominator = 1.0 + gross_return
    pretrade = [weight * value / denominator for weight, value in zip(target, gross)]
with (output / "scored_panel.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=panel_fields)
    writer.writeheader()
    writer.writerows(rows)

methods = ["shrinkage_mean_variance", "dcc_garch", "static_vine",
           "rolling_vine", "dynamic_nn_vine"]
with (output / "solver_audit.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=["method", "decision_date", "convergence"])
    writer.writeheader()
    for method in methods:
        for row in source:
            writer.writerow({"method": method, "decision_date": row["decision_date"],
                             "convergence": 1})
'''


def example_contract() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_contract(tmp_path: Path, runner_source: str = RUNNER_SOURCE) -> tuple[dict, Path]:
    contract = example_contract()
    contract["environment_lock"]["python_version"] = platform.python_version()
    contract["environment_lock"]["r_version"] = "test-r-1.0"
    contract["environment_lock"]["platform"] = "test-platform"
    contract["execution"]["access_ledger_path"] = "access/FUTURE_TEST_ACCESS_STARTED.json"
    for command in contract["execution"]["commands"]:
        command["argv"][0] = "python"

    by_id = {
        row["artifact_id"]: row
        for category in contract["artifacts"].values()
        for row in category
    }
    runner = tmp_path / by_id["confirmatory_runner"]["path"]
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text(runner_source, encoding="utf-8")

    for window in contract["windows"]:
        test_row = by_id[window["test_data_artifact_id"]]
        path = tmp_path / test_row["path"]
        assets = contract["assets"]
        rows = []
        for period in window["periods"]:
            row = {"window_id": window["window_id"], **period}
            row.update({f"g_{asset}": 1.0 for asset in assets})
            rows.append(row)
        write_csv(
            path,
            ["window_id", "decision_date", "holding_end_date",
             *[f"g_{asset}" for asset in assets]],
            rows,
        )

    attestation = tmp_path / by_id["runtime_attestation"]["path"]
    attestation.parent.mkdir(parents=True, exist_ok=True)
    attestation.write_text(json.dumps({
        "python_version": contract["environment_lock"]["python_version"],
        "r_version": contract["environment_lock"]["r_version"],
        "platform": contract["environment_lock"]["platform"],
        "container_image_digest": contract["environment_lock"]["container_image_digest"],
    }), encoding="utf-8")

    for category, rows in contract["artifacts"].items():
        for row in rows:
            path = tmp_path / row["path"]
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"frozen {category} artifact {row['artifact_id']}\n", encoding="utf-8")
            row["sha256"] = sha256_file(path)

    contract_path = tmp_path / "future_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract, contract_path


def test_example_contract_passes_schema_only_validation() -> None:
    report = validate_contract_schema(example_contract())
    assert report["window_count"] == 2
    assert report["asset_count"] == 7
    assert report["future_test_start"] == "2026-07-31"


def test_future_contract_requires_actual_elapsed_annualization() -> None:
    contract = example_contract()
    assert contract["economics"]["annualization_convention"] == (
        "actual_elapsed_years_v1"
    )
    contract["economics"]["annualization_convention"] = (
        "fixed_periods_per_year_v1"
    )
    with pytest.raises(ProtocolError, match="annualization_convention"):
        validate_contract_schema(contract)


def test_new_test_windows_cannot_overlap_consumed_holdout() -> None:
    contract = example_contract()
    contract["consumed_holdouts"][0]["end"] = "2026-08-01"
    with pytest.raises(ProtocolError, match="overlaps the consumed holdout"):
        validate_contract_schema(contract)


def test_training_validation_test_chronology_is_fail_closed() -> None:
    contract = example_contract()
    contract["windows"][0]["validation_end"] = "2026-08-15"
    with pytest.raises(ProtocolError, match="chronology"):
        validate_contract_schema(contract)


def test_test_command_cannot_consume_validation_data() -> None:
    contract = example_contract()
    command = contract["execution"]["commands"][0]
    command["argv"].append("{artifact:wf01_validation_data}")
    command["input_artifact_ids"].append("wf01_validation_data")
    with pytest.raises(ProtocolError, match="validation data during test"):
        validate_contract_schema(contract)


def test_full_validation_detects_separate_artifact_hash_tampering(tmp_path: Path) -> None:
    contract, _ = materialize_contract(tmp_path)
    validate_full_contract(contract, tmp_path)
    model = tmp_path / contract["artifacts"]["model"][0]["path"]
    model.write_text("tampered model", encoding="utf-8")
    with pytest.raises(ProtocolError, match="model artifact hash mismatch"):
        validate_full_contract(contract, tmp_path)


def write_solver_audit(path: Path, contract: dict, window: dict, code: int) -> None:
    rows = [
        {"method": method, "decision_date": period["decision_date"], "convergence": code}
        for method in contract["solver"]["methods"]
        for period in window["periods"]
    ]
    write_csv(path, ["method", "decision_date", "convergence"], rows)


def test_solver_code_five_is_rejected_without_fallback(tmp_path: Path) -> None:
    contract = example_contract()
    audit = tmp_path / "solver_audit.csv"
    write_solver_audit(audit, contract, contract["windows"][0], code=5)
    with pytest.raises(ProtocolError, match="Rejected solver convergence code 5"):
        validate_solver_audit(audit, contract, contract["windows"][0])


def audited_panel_rows(contract: dict, window: dict) -> tuple[list[str], list[dict]]:
    assets = contract["assets"]
    economics = contract["economics"]
    target = [0.5, 0.4, 0.2, 0.1, 0.0, 0.0, -0.2]
    pretrade = [1.0 / len(assets)] * len(assets)
    rows: list[dict] = []
    for index, period in enumerate(window["periods"]):
        asset_gross = [1.02, 0.98, 1.01, 1.0, 0.99, 1.005, 0.995]
        if index % 2:
            asset_gross = list(reversed(asset_gross))
        decision = dt.date.fromisoformat(period["decision_date"])
        holding_end = dt.date.fromisoformat(period["holding_end_date"])
        days = (holding_end - decision).days
        turnover = sum(abs(left - right) for left, right in zip(target, pretrade))
        short = sum(max(-weight, 0.0) for weight in target)
        cash = max(sum(target) - 1.0, 0.0)
        transaction = economics["turnover_cost"] * turnover
        financing = days / economics["day_count_basis"] * (
            economics["annual_short_borrow_rate"] * short
            + economics["annual_cash_borrow_rate"] * cash
        )
        gross_return = sum(weight * gross for weight, gross in zip(target, asset_gross)) - 1
        net_return = math.exp(math.log1p(gross_return) - transaction - financing) - 1
        row = {
            "strategy_id": "test_strategy", "window_id": window["window_id"],
            **period, "calendar_days": days, "gross_return": gross_return,
            "net_return": net_return, "turnover": turnover,
            "transaction_cost": transaction, "financing_cost": financing,
            "short_notional": short, "cash_borrow_notional": cash,
            "gross_exposure": sum(abs(weight) for weight in target),
            "net_exposure": sum(target),
        }
        row.update({f"target_w_{asset}": value for asset, value in zip(assets, target)})
        row.update({f"pretrade_w_{asset}": value for asset, value in zip(assets, pretrade)})
        row.update({f"g_{asset}": value for asset, value in zip(assets, asset_gross)})
        rows.append(row)
        denominator = 1 + gross_return
        pretrade = [weight * gross / denominator for weight, gross in zip(target, asset_gross)]
    fields = [
        "strategy_id", "window_id", "decision_date", "holding_end_date",
        "calendar_days", "gross_return", "net_return", "turnover",
        "transaction_cost", "financing_cost", "short_notional",
        "cash_borrow_notional", "gross_exposure", "net_exposure",
        *[f"target_w_{asset}" for asset in assets],
        *[f"pretrade_w_{asset}" for asset in assets],
        *[f"g_{asset}" for asset in assets],
    ]
    return fields, rows


def test_panel_enforces_drift_turnover_and_partial_financing(tmp_path: Path) -> None:
    contract = example_contract()
    window = contract["windows"][0]
    fields, rows = audited_panel_rows(contract, window)
    panel = tmp_path / "panel.csv"
    write_csv(panel, fields, rows)
    validate_evaluation_panel(panel, contract, window)

    bad_turnover = copy.deepcopy(rows)
    bad_turnover[1]["turnover"] += 0.01
    write_csv(panel, fields, bad_turnover)
    with pytest.raises(ProtocolError, match="drift-aware turnover"):
        validate_evaluation_panel(panel, contract, window)

    bad_financing = copy.deepcopy(rows)
    bad_financing[0]["financing_cost"] *= 30 / 31
    write_csv(panel, fields, bad_financing)
    with pytest.raises(ProtocolError, match="prorated financing cost"):
        validate_evaluation_panel(panel, contract, window)


def test_locked_execution_is_successful_immutable_and_single_access(tmp_path: Path) -> None:
    _, contract_path = materialize_contract(tmp_path)
    output = tmp_path / "immutable_results"
    bundle = tmp_path / "immutable_results.tar.gz"
    manifest = execute_contract(contract_path, tmp_path, output, bundle)
    assert manifest["status"] == "complete"
    assert (output / "future_confirmatory_manifest.json").is_file()
    assert bundle.is_file()
    assert bundle.with_suffix(bundle.suffix + ".sha256").is_file()
    assert (tmp_path / "access/FUTURE_TEST_ACCESS_STARTED.json").is_file()
    with pytest.raises(ProtocolError, match="Immutable output already exists"):
        execute_contract(contract_path, tmp_path, output, bundle)


def test_accessed_solver_failure_is_archived_and_not_retryable(tmp_path: Path) -> None:
    bad_runner = RUNNER_SOURCE.replace('"convergence": 1', '"convergence": 5')
    _, contract_path = materialize_contract(tmp_path, runner_source=bad_runner)
    output = tmp_path / "failed_results"
    bundle = tmp_path / "failed_results.tar.gz"
    with pytest.raises(ProtocolError, match="Rejected solver convergence code 5"):
        execute_contract(contract_path, tmp_path, output, bundle)
    failure = json.loads((output / "future_confirmatory_manifest.json").read_text())
    assert failure["status"] == "failed"
    assert failure["test_access_started"] is True
    assert bundle.is_file()
    with pytest.raises(ProtocolError):
        execute_contract(contract_path, tmp_path, tmp_path / "different_output",
                         tmp_path / "different_output.tar.gz")


def test_tar_failure_preserves_accessed_logs_and_failure_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, contract_path = materialize_contract(tmp_path)
    output = tmp_path / "tar_failed_results"
    bundle = tmp_path / "tar_failed_results.tar.gz"

    def fail_tar(*_args, **_kwargs) -> None:
        raise OSError("simulated compressor failure")

    monkeypatch.setattr(protocol_module, "deterministic_tar", fail_tar)
    with pytest.raises(ProtocolError, match="Failure logs were preserved"):
        execute_contract(contract_path, tmp_path, output, bundle)

    failure = json.loads((output / "future_confirmatory_manifest.json").read_text())
    assert failure["status"] == "failed"
    assert failure["test_access_started"] is True
    assert "simulated compressor failure" in failure["error"]
    assert any((output / "command_logs").glob("*.stdout.txt"))
    assert not bundle.exists()
    assert not bundle.with_suffix(bundle.suffix + ".sha256").exists()
    assert (tmp_path / "access/FUTURE_TEST_ACCESS_STARTED.json").is_file()
