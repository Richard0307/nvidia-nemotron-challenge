#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/sft_baseline.yaml}"
DRY_RUN="${DRY_RUN:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
VENV_DIR="${VENV_DIR:-.venv}"

ARGS=(train.py --config "$CONFIG")

if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi

if [[ "$MAX_SAMPLES" != "0" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi

"$VENV_DIR/bin/python" "${ARGS[@]}"
