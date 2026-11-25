#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
source_dir="$root/vendor/llama.cpp"
revision=5220a2d4040aa75f52678ebf52d77a0fba85de9f
patch="$root/patches/llama.cpp.patch"
mkdir -p vendor bin
if [[ ! -e "$source_dir/.git" ]]; then
    git init -q "$source_dir"
    git -C "$source_dir" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
if ! git -C "$source_dir" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "$source_dir" fetch --depth=1 origin "$revision"
    git -C "$source_dir" checkout --detach "$revision"
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]]; then
    echo "Expected llama.cpp revision $revision in $source_dir" >&2
    exit 1
fi
if ! git -C "$source_dir" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$source_dir" apply "$patch"
fi
cmake -S "$source_dir" -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build build --target llama-server llama-completion --parallel "${JOBS:-6}"
cp build/bin/llama-server build/bin/llama-completion bin/
