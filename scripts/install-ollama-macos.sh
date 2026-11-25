#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -s)" == Darwin ]] || { echo "This installer requires macOS." >&2; exit 1; }
mkdir -p bin
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
archive="$scratch/ollama-darwin.tgz"
curl -fLsS --retry 3 -o "$archive" https://github.com/ollama/ollama/releases/download/v0.40.18/ollama-darwin.tgz
echo "015261f434d31d7f0781f8ba6b59cc8e1f70ecb562a4e6db44dfdd5e02f7e562  $archive" | shasum -a 256 -c -
tar -xzf "$archive" -C "$scratch" ollama
install -m 755 "$scratch/ollama" bin/ollama
