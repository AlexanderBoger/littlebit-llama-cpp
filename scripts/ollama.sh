#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
export OLLAMA_HOST="${LITTLEBIT_HOST:-127.0.0.1:14679}"
export OLLAMA_MODELS="$root/models/ollama"
export OLLAMA_NO_CLOUD=1
exec "$root/bin/ollama" "$@"
