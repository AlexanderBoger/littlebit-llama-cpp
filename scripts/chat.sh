#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
exec "$root/bin/llama-completion" -m "$root/models/littlebit-qwen3.gguf" \
    -ngl 0 -c 2048 --conversation --temp 0.6 --top-k 20 --top-p 0.95 "$@"
