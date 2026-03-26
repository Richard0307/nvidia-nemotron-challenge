#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <adapter_dir> [output_zip]" >&2
  exit 1
fi

ADAPTER_DIR="$1"
OUTPUT="${2:-}"
VENV_DIR="${VENV_DIR:-.venv}"

ARGS=(package_submission.py --adapter-dir "$ADAPTER_DIR")

if [[ -n "$OUTPUT" ]]; then
  ARGS+=(--output "$OUTPUT")
fi

"$VENV_DIR/bin/python" "${ARGS[@]}"
