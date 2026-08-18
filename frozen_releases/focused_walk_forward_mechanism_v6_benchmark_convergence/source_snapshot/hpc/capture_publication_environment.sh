#!/usr/bin/env bash
# Capture the R, orchestration-Python, policy-Python, training-Python, OS, and accelerator
# environment used for publication evaluation/secondary experiments.
#
# This is an inventory, not proof that an earlier run used the same runtime.
# The metadata deliberately records whether the capture is retrospective.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd -P)}"
OUT_DIR="${ENV_MANIFEST_DIR:-$REPO_ROOT/provenance_environment_v4}"
RSCRIPT="${RSCRIPT:-/gabirel/miniforge3/bin/Rscript}"
PYTHON="${PYTHON:-/gabirel/miniforge3/bin/python3}"
CONDA="${CONDA:-/gabirel/miniforge3/bin/conda}"
: "${POLICY_PYTHON:?Set POLICY_PYTHON to the isolated policy-inference interpreter}"
: "${TRAIN_PYTHON:?Set TRAIN_PYTHON to the GPU-enabled training interpreter}"
EXPECTED_TRAIN_GPUS="${EXPECTED_TRAIN_GPUS:-1}"
export EXPECTED_TRAIN_GPUS
CAPTURE_TIMING="${CAPTURE_TIMING:-retrospective_after_locked_oos_v4}"

cd "$REPO_ROOT"
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC

if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite environment evidence: $OUT_DIR" >&2
  exit 2
fi
for executable in "$RSCRIPT" "$PYTHON" "$POLICY_PYTHON" "$TRAIN_PYTHON"; do
  if [[ ! -x "$executable" ]]; then
    echo "Required executable is missing or not executable: $executable" >&2
    exit 2
  fi
done

mkdir -p "$OUT_DIR"

cat >"$OUT_DIR/capture_metadata.json" <<EOF
{
  "schema_version": 2,
  "capture_purpose": "publication_provenance_runtime_inventory",
  "capture_timing": "$CAPTURE_TIMING",
  "historical_identity_claimed": false,
  "prospective_for_following_release": true,
  "scientific_note": "This inventory is exact for the capture time and may attest only experiments launched from the immediately following immutable release; it is not proof of any earlier run environment."
}
EOF

{
  echo "captured_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root=$REPO_ROOT"
  echo "kernel=$(uname -a)"
  echo "rscript=$RSCRIPT"
  echo "python=$PYTHON"
  echo "policy_python=$POLICY_PYTHON"
  echo "training_python=$TRAIN_PYTHON"
  echo "locale:"
  locale
  if command -v lscpu >/dev/null 2>&1; then
    echo "lscpu:"
    lscpu
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi:"
    nvidia-smi -q
  else
    echo "nvidia-smi=unavailable"
  fi
} >"$OUT_DIR/system_runtime.txt"

"$RSCRIPT" --version >"$OUT_DIR/r_version.txt" 2>&1
"$RSCRIPT" --vanilla -e 'sessionInfo()' >"$OUT_DIR/r_session_info.txt" 2>&1
R_PACKAGE_MANIFEST="$OUT_DIR/r_installed_packages.csv" \
  "$RSCRIPT" --vanilla -e \
  'p <- as.data.frame(installed.packages()[, c("Package", "Version", "LibPath"), drop = FALSE]); p <- p[order(p$Package, p$Version, p$LibPath), ]; write.csv(p, Sys.getenv("R_PACKAGE_MANIFEST"), row.names = FALSE)'

"$PYTHON" - <<'PY' >"$OUT_DIR/orchestration_python_runtime.json"
import json
import os
import platform
import sys

payload = {
    "executable": sys.executable,
    "implementation": platform.python_implementation(),
    "platform": platform.platform(),
    "python_version": platform.python_version(),
}
for package in ("numpy", "pandas", "scipy", "matplotlib", "seaborn", "yaml"):
    try:
        module = __import__(package)
        payload[package] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        payload[package] = f"unavailable: {type(exc).__name__}: {exc}"
print(json.dumps(payload, indent=2, sort_keys=True))
PY
"$PYTHON" -m pip freeze --all >"$OUT_DIR/orchestration_python_pip_freeze.txt"

"$POLICY_PYTHON" - <<'PY' >"$OUT_DIR/policy_python_runtime.json"
import json
import platform
import sys

import torch

payload = {
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_runtime": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "executable": sys.executable,
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "torch_path": torch.__file__,
    "torch_version": torch.__version__,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
"$POLICY_PYTHON" -m pip freeze --all >"$OUT_DIR/policy_python_pip_freeze.txt"

"$TRAIN_PYTHON" - <<'PY' >"$OUT_DIR/training_python_runtime.json"
import json
import os
import platform
import sys

import torch

payload = {
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()),
    "cuda_device_names": [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ],
    "cuda_runtime": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "executable": sys.executable,
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "torch_path": torch.__file__,
    "torch_version": torch.__version__,
}
expected = int(os.environ.get("EXPECTED_TRAIN_GPUS", "1"))
payload["minimum_required_gpu_count"] = expected
if not payload["cuda_available"] or payload["cuda_device_count"] < expected:
    raise SystemExit(
        f"Training Python exposes {payload['cuda_device_count']} GPUs; {expected} required."
    )
print(json.dumps(payload, indent=2, sort_keys=True))
PY
"$TRAIN_PYTHON" -m pip freeze --all >"$OUT_DIR/training_python_pip_freeze.txt"

if [[ -x "$CONDA" ]]; then
  "$CONDA" info --json >"$OUT_DIR/conda_info.json"
  "$CONDA" list --explicit >"$OUT_DIR/orchestration_conda_explicit_linux-64.txt"
  "$CONDA" env export >"$OUT_DIR/orchestration_conda_environment_full.yml"

  training_prefix="$(cd "$(dirname "$TRAIN_PYTHON")/.." && pwd -P)"
  if [[ -d "$training_prefix/conda-meta" ]]; then
    "$CONDA" list --prefix "$training_prefix" --explicit \
      >"$OUT_DIR/training_conda_explicit_linux-64.txt"
    "$CONDA" env export --prefix "$training_prefix" \
      >"$OUT_DIR/training_conda_environment_full.yml"
  else
    printf 'training Python prefix is not managed by conda: %s\n' \
      "$training_prefix" >"$OUT_DIR/training_conda_unavailable.txt"
  fi
else
  printf 'conda executable unavailable: %s\n' "$CONDA" \
    >"$OUT_DIR/conda_unavailable.txt"
fi

(
  cd "$OUT_DIR"
  find . -maxdepth 1 -type f ! -name CONTENTS.sha256 -print0 |
    sort -z |
    xargs -0 sha256sum >CONTENTS.sha256
)

echo "Environment evidence captured at: $OUT_DIR"
echo "Capture timing recorded in capture_metadata.json: $CAPTURE_TIMING"
