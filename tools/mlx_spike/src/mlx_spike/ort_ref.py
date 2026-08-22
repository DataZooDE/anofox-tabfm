"""The CPU reference. ORT on CPU is the definition of correct for every
comparison in this project, so this module is the one place that decides what
"the right answer" is.

The ext graph carries its initializers as ONNX external data with `location =
"model.safetensors"` and absolute byte offsets -- the same trick the C++ engine
uses to let ORT read 300 MB straight off disk instead of copying it. ORT
resolves that location relative to the model file's own directory, so the graph
has to sit next to the weights. We stage a symlink rather than copying either.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import MITRA_CACHE, RESOURCES, SpikeError, Workload, load_mitra_weights


def stage_ext_graph(task: str = "classification") -> Path:
    """Place the ext graph beside the weights it indexes; return its path."""
    weights_dir = MITRA_CACHE[task]
    load_mitra_weights(task)  # validates the header sha before we trust offsets
    src = RESOURCES / f"graph_ext_mitra_{task}.onnx"
    if not src.exists():
        raise SpikeError(f"bundled graph missing: {src}")
    dst = weights_dir / src.name
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())
    return dst


class OrtReference:
    """ORT CPU session over the mitra ext graph."""

    def __init__(self, task: str = "classification", *, intra_threads: int | None = None):
        import onnxruntime as ort

        self.task = task
        self.graph_path = stage_ext_graph(task)
        opts = ort.SessionOptions()
        if intra_threads is not None:
            opts.intra_op_num_threads = intra_threads
        self.session = ort.InferenceSession(
            str(self.graph_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        served_by = self.session.get_providers()
        # A reference that quietly ran somewhere else is not a reference.
        if served_by != ["CPUExecutionProvider"]:
            raise SpikeError(f"ORT reference must run on CPU alone, got {served_by}")
        self.served_by = served_by[0]

    def run(self, w: Workload) -> np.ndarray:
        feeds = {
            "x": w.x.astype(np.float32),
            "y": w.y.astype(np.float32),
            "train_size": np.array([w.train_size], dtype=np.int64),
            "d": np.array([w.d], dtype=np.int64),
        }
        (logits,) = self.session.run(["logits"], feeds)
        return logits
