#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-data/train.csv}"
OUTPUT="${2:-data/train_cot.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
VENV_DIR="${VENV_DIR:-.venv}"

ARGS=(generate_cot.py --input "$INPUT" --output "$OUTPUT")

if [[ "$MAX_SAMPLES" != "0" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi

"$VENV_DIR/bin/python" "${ARGS[@]}"
