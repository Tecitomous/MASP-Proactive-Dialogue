#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
python - <<'PY'
import importlib

for name in ("masp", "masp.mind.bdi_schema", "masp.eval.metrics"):
    importlib.import_module(name)
    print(f"ok: {name}")
PY
python -m compileall -q src scripts
echo "MASP installation checks passed"
