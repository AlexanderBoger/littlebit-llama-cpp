"""Compare first-token distributions with Transformers and Samsung's LittleBitLinear."""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3RotaryEmbedding


PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4,",
    "Invoice total: EUR 129.50. Currency:",
]


def reference_model(checkpoint, weights, samsung):
    sys.path.insert(0, str(samsung.resolve()))
    # Initialise the upstream package in the same order as its evaluation loader.
    from quantization.utils.quant_util import LittleBitLinear
    from quantization.utils.binary_packer import binary_unpacker

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    model.model.rotary_emb = Qwen3RotaryEmbedding(config)

    class CheckpointProjection(torch.nn.Module):
        def __init__(self, prefix, outputs):
            super().__init__()
            self.prefix = prefix
            self.outputs = outputs

        def forward(self, x):
            # Unpack one projection at a time to keep reference RAM manageable.
            layer = LittleBitLinear()
            layer.out_features = self.outputs
            layer.residual = config.residual
            layer.bias = None
            layer._binarized = True
            for suffix in (("", "_R") if layer.residual else ("",)):
                for factor in ("U", "V"):
                    key = self.prefix + "." + factor + suffix
                    shape = weights.get_tensor(key + "_shape").tolist()
                    signs = binary_unpacker(weights.get_tensor(key + "_packed"), tuple(shape))
                    setattr(layer, factor + suffix, signs)
                for scale in ("u1", "u2", "v1", "v2"):
                    name = scale + suffix
                    setattr(layer, name, weights.get_tensor(self.prefix + "." + name))
            return layer(x)

    keys = set(weights.keys())
    for name, module in list(model.named_modules()):
        if name + ".V_packed" in keys:
            model.set_submodule(name, CheckpointProjection(name, module.out_features))
    for name, _ in list(model.named_parameters()):
        owner, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(owner), attr, torch.nn.Parameter(weights.get_tensor(name), requires_grad=False))
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("samsung", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:14680")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    batch = tokenizer(PROMPTS, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
    with safe_open(args.checkpoint / "model.safetensors", framework="pt") as weights, torch.inference_mode():
        model = reference_model(args.checkpoint, weights, args.samsung)
        torch.set_num_threads(6)
        logits = model(**batch).logits.float()
    passed = True
    for i, prompt in enumerate(PROMPTS):
        length = int(batch.attention_mask[i].sum())
        ids = batch.input_ids[i, :length].tolist()
        reference = logits[i, length - 1].log_softmax(0)
        top = reference.topk(10).indices.tolist()
        payload = {"prompt": ids, "n_predict": 1, "temperature": 0, "n_probs": 10, "cache_prompt": False}
        request = urllib.request.Request(args.server + "/completion", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            native = json.load(response)["completion_probabilities"][0]["top_logprobs"]
        native_ids = [item["id"] for item in native]
        overlap = len(set(top) & set(native_ids))
        difference = max(abs(item["logprob"] - reference[item["id"]].item()) for item in native)
        print(f"{prompt!r}: reference {top[0]}, native {native_ids[0]}, top-10 overlap {overlap}/10, "
              f"maximum log-probability difference {difference:.3f}")
        passed &= overlap >= 7 and native_ids[0] in top[:3] and difference < 1.0
    if not passed:
        raise SystemExit("Native distributions exceeded the regression tolerances")


if __name__ == "__main__":
    main()
