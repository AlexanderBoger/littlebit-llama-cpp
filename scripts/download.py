"""Download only the inference files of the pinned public community checkpoint."""

import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    dest = Path(__file__).resolve().parents[1] / "models/checkpoint"
    snapshot_download(
        repo_id="LittleBitLLM/littlebit-qwen3-8b-0.55bpw",
        revision="6b2c47aa9707e624e2e737d16d64a66be8090ae9",
        local_dir=dest, allow_patterns=["*.safetensors", "*.json", "merges.txt"],
    )
    with (dest / "model.safetensors").open("rb") as weights:
        digest = hashlib.file_digest(weights, "sha256").hexdigest()
    if digest != "5b087e4c79c23697355014fdd00b083c91d0c731ddd80a16d6affbe156f17179":
        raise SystemExit(f"Unexpected SHA-256 for model.safetensors: {digest}")
    print(f"Verified the checkpoint at {dest}")
