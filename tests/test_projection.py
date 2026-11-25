import unittest

import numpy as np
import torch

from convert import export_branch


class ProjectionTest(unittest.TestCase):
    def test_scaled_main_and_residual_with_rank_padding(self):
        rng = np.random.default_rng(42)
        rank, inputs, outputs = 40, 128, 16
        x = rng.normal(size=(2, inputs)).astype("f")
        expected = np.zeros((2, outputs))
        actual = np.zeros_like(expected)
        for suffix in ("", "_R"):
            tensors, gguf = {}, {}
            factors, scales = {}, {}
            for name, dims in (("V", (rank, inputs)), ("U", (outputs, rank))):
                signs = rng.choice([-1, 1], size=dims).astype("f")
                padded = np.pad(signs < 0, ((0, 0), (0, -dims[1] % 32)))
                words = np.packbits(padded, axis=1, bitorder="little").view("<i4")
                tensors["p." + name + suffix + "_packed"] = torch.from_numpy(words)
                tensors["p." + name + suffix + "_shape"] = torch.tensor(dims)
                factors[name] = signs
            for key, n in (("u1", outputs), ("u2", rank), ("v1", rank), ("v2", inputs)):
                scales[key] = torch.from_numpy(rng.normal(size=n)).bfloat16()
                tensors["p." + key + suffix] = scales[key].unsqueeze(0)

            def write(name, data, binary):
                if binary:
                    blocks = data.reshape(data.shape[0], -1, 18)
                    bits = np.unpackbits(blocks[:, :, 2:], axis=2, bitorder="little")
                    data = bits.astype("f").reshape(data.shape[0], -1) * 2 - 1
                gguf[name] = data

            export_branch(tensors.pop, "p", suffix, write, (inputs, outputs))
            self.assertFalse(tensors)
            middle = (scales["v1"] * scales["u2"]).float().numpy()
            h = (x * scales["v2"].float().numpy()) @ factors["V"].T
            expected += ((h * middle) @ factors["U"].T) * scales["u1"].float().numpy()
            h = (x * gguf["in"]) @ gguf["v"].T
            actual += ((h * gguf["mid"]) @ gguf["u"].T) * gguf["out"]

        np.testing.assert_allclose(actual, expected, atol=0.0001, rtol=0.00001)
