"""Shared pieces for the MLX spikes: deterministic inputs, weight loading.

The input generator must match make_golden.py EXACTLY -- the golden file
records its own sha of the inputs so a drifted generator fails loudly
instead of comparing apples to oranges.
"""
import hashlib
import json
import os
import struct

import numpy as np

T, H, TRAIN, D_CLASSES = 90, 3, 60, 3


def make_inputs():
    rng = np.random.default_rng(20260822)
    x = rng.standard_normal((1, T, H)).astype(np.float32)
    y = np.zeros((1, T), dtype=np.float32)
    y[0, :TRAIN] = rng.integers(0, D_CLASSES, TRAIN)
    feed = {
        "x": x,
        "y": y,
        "train_size": np.array([TRAIN], np.int64),
        "d": np.array([H], np.int64),
    }
    sha = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    return feed, sha


def download_mitra(dest_dir):
    """Fetch the public mitra classifier weights (Apache-2.0, no credentials)."""
    from huggingface_hub import hf_hub_download
    os.makedirs(dest_dir, exist_ok=True)
    return hf_hub_download("autogluon/mitra-classifier", "model.safetensors",
                           local_dir=dest_dir)


def load_safetensors_f32(path):
    """{key: np.float32 array} straight from the file -- no torch needed."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        out = {}
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            assert meta["dtype"] == "F32", (key, meta["dtype"])
            b, e = meta["data_offsets"]
            f.seek(base + b)
            out[key] = np.frombuffer(f.read(e - b), dtype=np.float32).reshape(meta["shape"]).copy()
    return out
