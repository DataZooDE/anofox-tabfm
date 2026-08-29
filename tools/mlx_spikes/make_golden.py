"""Generate the golden reference on the Linux box: ORT CPU + real weights.

Run from tools/mlx_spikes with the repo's export_mitra env (has onnx +
onnxruntime):  uv run --project ../export_mitra python make_golden.py
"""
import json
import os

import numpy as np
import onnxruntime as ort

from common import make_inputs

CACHE = os.path.expanduser("~/.cache/anofox-tabfm/autogluon__mitra-classifier@main")
GRAPH = os.path.join(CACHE, "graph_ext_mitra_classification.onnx")  # staged beside weights

feed, input_sha = make_inputs()
sess = ort.InferenceSession(GRAPH, providers=["CPUExecutionProvider"])
names = [i.name for i in sess.get_inputs()]
logits = sess.run(["logits"], {k: v for k, v in feed.items() if k in names})[0]

out = {
    "input_sha256": input_sha,
    "logits_shape": list(logits.shape),
    "logits": logits.astype(float).round(6).tolist(),
    "note": "ORT CPU, real autogluon/mitra-classifier weights, inputs from common.make_inputs()",
}
with open("golden_mitra_classification.json", "w") as f:
    json.dump(out, f)
print(f"MARKER:GOLDEN_WRITTEN shape={logits.shape} input_sha={input_sha[:12]}")
