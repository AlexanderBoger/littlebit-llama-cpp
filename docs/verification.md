# Verification on November 25, 2026

The test machine was an Apple M5 Pro with 24 GB of unified memory. The runtime used six CPU threads, a 512-token context and the community Qwen3-8B LittleBit-2 checkpoint. The downloaded safetensors file matched the publisher's SHA-256, `5b087e4c79c23697355014fdd00b083c91d0c731ddd80a16d6affbe156f17179`.

## Standalone release validation

A fresh checkout built the CPU runner and independently converted the pinned checkpoint. The three unit tests passed. The 3,098,366,336-byte GGUF matched SHA-256 `c29a3408b1375042b989f1f65aa224ebe48a22f9ea4444cf292a8ea25a147896`. The macOS and Linux ARM64 Docker builds each imported this file into an isolated Ollama 0.40.18 model store and produced the same 12-token greedy completion. The Linux build ran inside Docker Desktop on the same Apple Silicon machine. The reference parity table below was reproduced with the final macOS build. Linux x86-64, Windows/WSL, GPU acceleration and iPhone execution were not tested.

## Output parity

The reference used Transformers Qwen3 and Samsung's original binary unpacker and LittleBitLinear forward pass. Each row compares the same prompt token IDs and the next-token distribution. The last column is the maximum absolute log-probability difference among the native top ten.

| Prompt | Reference / native top token ID | Top-ten overlap | Max difference |
| --- | --- | --- | --- |
| The capital of France is | 12095 / 12095 | 10/10 | 0.143 |
| 1, 2, 3, 4, | 220 / 220 | 8/10 | 0.118 |
| Invoice total: EUR 129.50. Currency: | 37088 / 37088 | 10/10 | 0.105 |

An independent check decoded the GGUF factors for the first attention key projection (rank 208) and feed-forward down projection (rank 824). The main and residual sign matrices matched Samsung's unpacker bit for bit. Their summed FP32 projection outputs also matched for the seeded inputs.

## Runtime and model behaviour

The llama.cpp completion binary and the Ollama `/api/generate` endpoint both continued the raw prompt “The capital of France is” with “ Paris, and the city is the capital of the French Republic” at temperature zero. Ollama imported the 3,098,366,336-byte GGUF and launched the custom native server from beside its own executable.

In one 32-token raw completion on this Mac, llama.cpp reported 11.10 generated tokens/s over 31 timed decode steps. macOS reported a maximum resident set size of 2,848,030,720 bytes (2.65 GiB). This measurement used memory mapping. It describes that process and short context only. To repeat, prefix the README's raw-completion command with `/usr/bin/time -l`, add `--no-warmup` and change `-n 12` to `-n 32`.

The document trial supplied fictional invoice text with a total of EUR 129.50. At temperature zero the chat endpoint returned whitespace. With sampling (0.6 temperature, seed 42) it repeated an unrelated number. The port runs this checkpoint, but these trials give no evidence that it can extract invoice amounts reliably. PDF field accuracy, performance on an 8 GB machine and cost per correctly extracted document remain unmeasured.
