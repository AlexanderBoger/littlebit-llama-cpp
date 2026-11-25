"""Export the packed Qwen3 LittleBit projections and the original Qwen3 tokenizer."""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from packing import BLOCK, encode_q1


def export_branch(take, stem, suffix, write, expected_dims):
    def get(key):
        return take(stem + "." + key)

    def shape(key):
        dims = get(key + suffix + "_shape")
        if dims.dtype != torch.int64 or dims.shape != (2,):
            raise ValueError(f"{stem}.{key}{suffix}_shape must contain two dimensions")
        return tuple(dims.tolist())

    def scale(key, length):
        value = get(key + suffix)
        if list(value.shape) not in ([1, length], [length]) or not value.is_floating_point():
            raise ValueError(f"{stem}.{key}{suffix} must contain {length} floating-point values")
        value = value.reshape(-1)
        if not torch.isfinite(value).all():
            raise ValueError(f"{stem}.{key}{suffix} contains nonfinite scales")
        return value

    rank, inputs = shape("V")
    outputs, rank_u = shape("U")
    if inputs % BLOCK or rank != rank_u or min(rank, inputs, outputs) < 1:
        raise ValueError(f"{stem} has unsupported factor dimensions {(rank, inputs)}, {(outputs, rank_u)}")
    if (inputs, outputs) != expected_dims:
        raise ValueError(f"{stem} has {(inputs, outputs)} input/output dimensions, expected {expected_dims}")
    padded = (rank + BLOCK - 1) // BLOCK * BLOCK
    for factor, dims, target in (("V", (rank, inputs), (padded, inputs)),
                                 ("U", (outputs, rank), (outputs, padded))):
        words = get(factor + suffix + "_packed").numpy()
        write(factor.lower(), encode_q1(words, dims, target), True)

    write("in", scale("v2", inputs).float().numpy(), False)
    # Match the checkpoint's dtype for this product, as in LittleBitLinear.forward.
    latent = (scale("v1", rank) * scale("u2", rank)).float().numpy()
    write("mid", np.pad(latent, (0, padded - rank)), False)
    write("out", scale("u1", outputs).float().numpy(), False)


def convert(checkpoint, outfile, llama_cpp):
    sys.path.insert(0, str(llama_cpp.resolve()))
    from conversion.base import LazyTorchTensor, gguf
    from conversion.qwen import Qwen3Model

    config = json.loads((checkpoint / "config.json").read_text())
    littlebit = json.loads((checkpoint / "littlebit_config.json").read_text())
    if config["architectures"] != ["Qwen3ForCausalLM"] or littlebit["quant_mod"] != "LittleBitLinear":
        raise ValueError("Expected a Qwen3ForCausalLM checkpoint with LittleBitLinear projections")
    if config.get("attention_bias") or config.get("mlp_bias"):
        raise ValueError("Only bias-free Qwen3 projections are supported")
    residual = littlebit["residual"]
    if not isinstance(residual, bool):
        raise ValueError("littlebit_config.json must specify residual as true or false")

    model = Qwen3Model(checkpoint, gguf.LlamaFileType.MOSTLY_BF16, outfile, use_temp_file=True,
                       model_name="LittleBit Qwen3")
    writer = model.gguf_writer
    writer.add_uint32("littlebit.version", 1)
    writer.add_bool("littlebit.residual", residual)

    def take(name):
        if name not in model.model_tensors:
            raise ValueError(f"Missing checkpoint tensor {name}")
        return LazyTorchTensor.to_eager(model.model_tensors.pop(name)())

    hidden = config["hidden_size"]
    ffn = config["intermediate_size"]
    head_size = config.get("head_dim", hidden // config["num_attention_heads"])
    queries = head_size * config["num_attention_heads"]
    kv = head_size * config["num_key_value_heads"]
    groups = {
        "self_attn.q_proj": (hidden, queries), "self_attn.k_proj": (hidden, kv),
        "self_attn.v_proj": (hidden, kv), "self_attn.o_proj": (queries, hidden),
        "mlp.gate_proj": (hidden, ffn), "mlp.up_proj": (hidden, ffn), "mlp.down_proj": (ffn, hidden),
    }
    for layer in range(config["num_hidden_layers"]):
        for group, dimensions in groups.items():
            stem = f"model.layers.{layer}.{group}"
            output = model.map_tensor_name(stem + ".weight").removesuffix(".weight")
            for suffix in (("", "_R") if residual else ("",)):
                prefix = output + (".lb_r_" if suffix else ".lb_")

                def write(field, data, binary):
                    writer.add_tensor(prefix + field, data, raw_dtype=gguf.GGMLQuantizationType.Q1_0 if binary else None)

                export_branch(take, stem, suffix, write, dimensions)
            for key in ("_eff_bit_actual", "_eff_bit_target", "_split_dim_final"):
                model.model_tensors.pop(stem + "." + key, None)
    model.write()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--llama-cpp", type=Path, default=Path(__file__).parent / "vendor/llama.cpp")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.output.suffix != ".gguf":
        parser.error("output must be a .gguf file")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as scratch:
        staging = Path(scratch) / args.output.name
        convert(args.checkpoint, staging, args.llama_cpp)
        staging.replace(args.output)


if __name__ == "__main__":
    main()
