import unittest

import numpy as np

from packing import encode_q1


class PackingTest(unittest.TestCase):
    def test_signed_words_and_partial_blocks(self):
        rng = np.random.default_rng(12)
        signs = rng.choice([-1, 1], size=(3, 136))
        source = np.packbits(signs < 0, axis=1, bitorder="little")
        words = np.pad(source, ((0, 0), (0, 3))).view("<i4")
        encoded = encode_q1(words, (3, 136), (4, 256)).reshape(4, 2, 18)
        np.testing.assert_array_equal(encoded[:, :, :2].copy().view("<f2"), 1)
        bits = np.unpackbits(encoded[:, :, 2:], axis=2, bitorder="little").reshape(4, 256)
        actual = 2 * bits.astype(int) - 1
        np.testing.assert_array_equal(actual[:3, :136], signs)

    def test_rejects_truncated_source(self):
        with self.assertRaises(ValueError):
            encode_q1(np.zeros((1, 4), dtype="<i4"), (1, 136), (1, 256))
