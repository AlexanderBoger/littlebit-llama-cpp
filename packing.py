"""Transcode LittleBit sign words to ggml Q1_0 without expanding the matrices."""

import numpy as np

BLOCK = 128


def encode_q1(words, shape, padded_shape):
    rows, cols = shape
    padded_rows, padded_cols = padded_shape
    if words.dtype != np.dtype("int32") or words.shape != (rows, (cols + 31) // 32):
        raise ValueError(f"Expected int32 sign words for {shape}, got {words.dtype} {words.shape}")
    if min(rows, cols) < 1 or padded_rows < rows or padded_cols < cols or padded_cols % BLOCK:
        raise ValueError(f"Invalid padded shape {padded_shape} for {shape}")

    blocks = padded_cols // BLOCK
    payload = np.zeros((padded_rows, blocks * 16), dtype=np.uint8)
    source = np.ascontiguousarray(words, dtype="<i4").view(np.uint8)
    # Samsung encodes positive signs as zero. ggml encodes them as one.
    payload[:rows, :source.shape[1]] = ~source
    encoded = np.empty((padded_rows, blocks, 18), dtype=np.uint8)
    encoded[:, :, :2] = np.array([1], dtype="<f2").view(np.uint8)
    encoded[:, :, 2:] = payload.reshape(padded_rows, blocks, 16)
    return encoded.reshape(padded_rows, -1)
