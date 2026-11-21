# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx", "onnxruntime", "numpy"]
# ///
"""Shape goldens for the Nori 1.5 graph, from ONNX Runtime (the oracle).

Needs the REAL Nori 1.5 ext graph and its safetensors (the weights are never written out; only tensor SHAPES are):

    uv run tools/ggml_parity/gen_nori15_shape_goldens.py <dir with graph_ext_nori15_regression.onnx + model.safetensors>

Adds a sample of the graph's intermediate tensors (every 5th node output, every node output that has five or more
dimensions, and every integer/bool node output) as extra graph outputs, runs ORT at several (rows, train, features)
combinations and writes test/fixtures/ggml_shapes/nori15_regression_T<rows>_S<train>_H<features>.shapes, one
`name d0,d1,...` line each; integer and bool tensors of at most 256 elements also carry their values after ` = `,
so the shape pass's folded integers are checked against ORT, not only its shapes.
The plain bundled graph has the same node and tensor names as the ext graph, so the goldens apply to it.
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

CONFIGS = [(60, 40, 9), (130, 100, 10), (24, 16, 3), (200, 150, 17), (9, 8, 1), (5, 1, 2)]
STEP = 5

src = sys.argv[1]
graph_path = os.path.join(src, "graph_ext_nori15_regression.onnx")
model = onnx.load(graph_path, load_external_data=False)
existing = {o.name for o in model.graph.output}
# Sequence-typed outputs (SplitToSequence) come back from ORT as lists, not tensors: not goldens.
sequence_outputs = {out for node in model.graph.node if node.op_type == "SplitToSequence" for out in node.output}
all_outs = [out for node in model.graph.node for out in node.output if out and out not in existing and out not in sequence_outputs]

# Pass 1 (tiny input, every node output): which tensors have five or more dimensions, which are integer/bool.
probe = onnx.load(graph_path, load_external_data=False)
for name in all_outs:
    probe.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
probe_path = os.path.join(src, "shape_probe.onnx")
onnx.save(probe, probe_path)
psess = ort.InferenceSession(probe_path, providers=["CPUExecutionProvider"])
prng = np.random.default_rng(1)
pouts = psess.run(all_outs, {"x": prng.random((1, 12, 5), dtype=np.float32), "y": prng.random((1, 8), dtype=np.float32)})
os.remove(probe_path)
index = {n: i for i, n in enumerate(all_outs)}
names = []
for i, node in enumerate(model.graph.node):
    for out in node.output:
        if not out or out in existing or out in sequence_outputs:
            continue
        arr = np.asarray(pouts[index[out]])
        if i % STEP == 0 or arr.ndim >= 5 or arr.dtype.kind in "iub":
            names.append(out)
for name in names:
    model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
dbg = os.path.join(src, "shape_dbg.onnx")
onnx.save(model, dbg)
sess = ort.InferenceSession(dbg, providers=["CPUExecutionProvider"])
out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "test", "fixtures", "ggml_shapes")
os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(0)
all_names = ["logits"] + names
for rows, train, feats in CONFIGS:
    x = (rng.random((1, rows, feats), dtype=np.float32) - 0.5)
    y = (rng.random((1, train), dtype=np.float32) - 0.5) * 100
    outs = sess.run(all_names, {"x": x, "y": y})
    lines = []
    for n, o in zip(all_names, outs):
        a = np.asarray(o)
        line = f"{n} {','.join(str(d) for d in a.shape)}"
        if a.dtype.kind in "iub" and a.size <= 256:
            line += " = " + ",".join(str(int(v)) for v in a.ravel())
        lines.append(line)
    path = os.path.join(out_dir, f"nori15_regression_T{rows}_S{train}_H{feats}.shapes")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(path, len(lines), "tensors")
os.remove(dbg)
