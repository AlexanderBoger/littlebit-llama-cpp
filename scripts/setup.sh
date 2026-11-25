#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.11 .venv
fi
export PATH="$PWD/.venv/bin:$PATH"
set --
if [[ "$(uname -s)" == Linux ]]; then
    set -- --torch-backend cpu
fi
uv pip install --python .venv/bin/python "$@" -r requirements.txt cmake ninja
scripts/build.sh
python -m unittest discover -v
python scripts/download.py
python convert.py models/checkpoint models/littlebit-qwen3.gguf
