# LittleBit for llama.cpp and Ollama

Run the community LittleBit-2 Qwen3-8B checkpoint on CPU. The converter writes its binary factors into a 3.10 GB GGUF. A patched llama.cpp runner evaluates the two factor matrices and learned scales directly. Ollama can use that runner with its regular CLI and HTTP API. This is an independent project.

## Get started

Install Git, a C++ compiler and [uv](https://docs.astral.sh/uv/getting-started/installation/). For the native macOS install, use Apple Silicon, macOS 14 or later, and the Xcode Command Line Tools. On Ubuntu or Windows with Ubuntu in WSL2, install the `build-essential` and `git` packages. Allow about 20 GB for dependencies, conversion and the local model store.

```sh
git clone https://github.com/AlexanderBoger/littlebit-llama-cpp.git
cd littlebit-llama-cpp
./scripts/setup.sh
./scripts/chat.sh
```

Setup creates a Python 3.11 environment, builds a pinned llama.cpp revision, runs the packing tests, downloads the public checkpoint and verifies its SHA-256 before conversion. To check raw completion independently of the chat template, run:

```sh
bin/llama-completion -m models/littlebit-qwen3.gguf -ngl 0 -t 6 -c 512 \
    --no-conversation --no-display-prompt --temp 0 -n 12 -p 'The capital of France is'
```

## Ollama

**On macOS**, install the pinned Ollama CLI next to the new runner. Start the project daemon in one terminal:

```sh
./scripts/install-ollama-macos.sh
./scripts/ollama.sh serve
```

In another terminal, import the model once and start a chat:

```sh
./scripts/ollama.sh create littlebit-qwen3 -f Modelfile
./scripts/ollama.sh run littlebit-qwen3
```

**With Docker**, use the converted GGUF from setup. Build the image on the machine that will serve it. The following commands compile a Linux CPU runner and place it where the bundled Ollama 0.40.18 looks for its inference backend. Run them in a POSIX shell, including WSL2 on Windows:

```sh
docker build -t littlebit-ollama .
docker run -d --name littlebit-ollama -p 127.0.0.1:14679:11434 \
    -v "$PWD/models/littlebit-qwen3.gguf:/opt/littlebit/models/littlebit-qwen3.gguf:ro" \
    -v littlebit-ollama-data:/root/.ollama littlebit-ollama
docker exec littlebit-ollama ollama create littlebit-qwen3 -f Modelfile
docker exec -it littlebit-ollama ollama run littlebit-qwen3
```

Both options expose an Ollama endpoint on `127.0.0.1:14679`. Existing clients can connect by setting `OLLAMA_HOST=127.0.0.1:14679`. For example:

```sh
curl -sS http://127.0.0.1:14679/api/generate -d '{
  "model":"littlebit-qwen3", "prompt":"The capital of France is",
  "raw":true, "stream":false,
  "options":{"temperature":0,"num_predict":12,"num_thread":6}
}'
```

## Supported model and format

The verified model is [LittleBitLLM/littlebit-qwen3-8b-0.55bpw](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw), revision `6b2c47aa9707e624e2e737d16d64a66be8090ae9`. The publisher's 0.55-bit figure applies to the linear layers. Embeddings and the output head remain BF16. Generic GGUF inspection can report 5.4B and BF16 because it counts the padded factor coordinates and reads the file-level weight type. The model card lists CC BY-NC 4.0 for the checkpoint. This repository distributes integration code and downloads weights directly from the publisher.

The converter maps sign words to ggml Q1_0 blocks and zeroes the latent scales for padding. In each projection the runner applies the input scale, the V matrix, the latent scale, the U matrix and the output scale. The released checkpoint uses a second branch with separate factors and scales. The runtime sums both contributions. Its CPU dot products quantize activations, so comparisons with PyTorch use numerical tolerances. [Samsung's reference implementation](https://github.com/SamsungLabs/LittleBit/blob/0ceadcfa842b2c26e14a6e00d98d47aed8e53dde/quantization/modules/littlebit.py)

The GGUF contains custom LittleBit metadata and requires the patched runner. The implemented architecture is bias-free dense Qwen3 with 128-aligned projection inputs. Other model architectures, LoRA, GPU execution and iOS are outside this release's verified scope. See the measured scope and known checkpoint limitations in the [validation results](docs/verification.md).

## Reproduce the reference check

Build and convert first, then start the project's native server in one terminal:

```sh
bin/llama-server -m models/littlebit-qwen3.gguf -ngl 0 -c 512 -np 1 -t 6 \
    --host 127.0.0.1 --port 14680 --no-webui
```

In a second terminal, compare next-token distributions for identical token IDs:

```sh
git clone https://github.com/SamsungLabs/LittleBit.git vendor/SamsungLittleBit
git -C vendor/SamsungLittleBit checkout --detach 0ceadcfa842b2c26e14a6e00d98d47aed8e53dde
.venv/bin/python scripts/verify_reference.py models/checkpoint vendor/SamsungLittleBit
```

## License

The original integration code is available under the [MIT license](LICENSE). The llama.cpp patch also carries the [ggml authors' MIT notice](patches/llama.cpp-LICENSE). The bundled Ollama CLI carries [Ollama's MIT notice](notices/ollama-LICENSE). The public checkpoint is a separate download under the publisher's [CC BY-NC 4.0 model license](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw).
