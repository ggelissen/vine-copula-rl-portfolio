from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from publication_pipeline_draft.register_frozen_artifact import (
    RegistrationError,
    register,
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    archive.addfile(info, io.BytesIO(value))


def make_release(tmp_path: Path, *, corrupt_internal: bool = False) -> tuple[Path, Path]:
    archive_path = tmp_path / "release.tar.gz"
    payload = b"evidence\n"
    manifest = json.dumps(
        {
            "release_status": "frozen_post_holdout_explanatory_ablation",
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claims_permitted": False,
        }
    ).encode()
    expected = "0" * 64 if corrupt_internal else digest(payload)
    inventory = f"{expected}  evidence.csv\n".encode()
    with tarfile.open(archive_path, "w:gz") as archive:
        root = "release"
        add_bytes(archive, f"{root}/READ_ONLY_RELEASE.txt", b"read only\n")
        add_bytes(
            archive,
            f"{root}/post_holdout_explanatory_release_manifest.json",
            manifest,
        )
        add_bytes(archive, f"{root}/CONTENTS.sha256", inventory)
        add_bytes(archive, f"{root}/evidence.csv", payload)
    sidecar = tmp_path / "release.tar.gz.sha256"
    sidecar.write_text(f"{digest(archive_path.read_bytes())}  {archive_path.name}\n")
    return archive_path, sidecar


def test_registers_exact_frozen_bytes(tmp_path: Path) -> None:
    archive, sidecar = make_release(tmp_path)
    output = tmp_path / "registry"
    result = register(archive, sidecar, output, registration_id="ablation_v2")
    assert result["operation"] == "byte_exact_registration_no_reexecution"
    assert result["verified_internal_files"] == 1
    assert (output / archive.name).read_bytes() == archive.read_bytes()
    assert not result["confirmatory_claims_permitted"]


def test_rejects_internal_checksum_failure(tmp_path: Path) -> None:
    archive, sidecar = make_release(tmp_path, corrupt_internal=True)
    with pytest.raises(RegistrationError, match="Internal checksum mismatch"):
        register(archive, sidecar, tmp_path / "registry", registration_id="bad")


def test_refuses_overwrite(tmp_path: Path) -> None:
    archive, sidecar = make_release(tmp_path)
    output = tmp_path / "registry"
    output.mkdir()
    with pytest.raises(RegistrationError, match="already exists"):
        register(archive, sidecar, output, registration_id="duplicate")
