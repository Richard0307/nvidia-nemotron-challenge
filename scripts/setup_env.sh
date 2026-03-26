#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

if [[ "$(uname -s)" == "Linux" ]]; then
  "$VENV_DIR/bin/pip" install -r requirements.linux.txt
fi

echo "Environment ready at $VENV_DIR"
