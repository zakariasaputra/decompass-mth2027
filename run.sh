#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_CSV="${1:-submission/Decompass.csv}"

"$PYTHON_BIN" run_pipeline.py --rebuild-cache --out "$OUT_CSV"
"$PYTHON_BIN" evaluation/validate_submission.py "$OUT_CSV"
