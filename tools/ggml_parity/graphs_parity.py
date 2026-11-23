# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx", "onnxruntime", "numpy"]
# ///
"""Whole-graph parity for every bundled CPU graph, ggml against ONNX Runtime, on deterministic pseudo-random weights.

    TABFM_GRAPHS_DUMP=/tmp/dump ./build/release/test/unittest "[ggml_graphs]"      # ggml runs, dumps inputs and outputs
    uv run tools/ggml_parity/graphs_parity.py /tmp/dump [graph-name ...]            # ORT runs the same graph on the same weights

The weights are never real (license wall): element i of the initializer `name` is hash(name, i) in [-0.1, 0.1), the formula of
TestWeights() in test/cpp/test_tabfm_ggml_graphs.cpp. Prepare/query pairs are chained exactly like the C++ test does.
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper

RESOURCES = os.path.join(os.path.dirname(__file__), "..", "..", "resources")


def test_weights(name, n):
    seed = np.uint32(2166136261)
    s = int(seed)
    for c in name.encode():
        s = ((s ^ c) * 16777619) & 0xFFFFFFFF
    i = np.arange(n, dtype=np.uint64)
    h = ((i * np.uint64(2654435761) + np.uint64(s)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return ((h >> 8).astype(np.float32) / np.float32(16777216.0) - np.float32(0.5)) * np.float32(0.2)


def load_with_weights(path):
    model = onnx.load(path, load_external_data=False)
    for init in model.graph.initializer:
        if init.data_location != TensorProto.EXTERNAL:
            continue
        if init.data_type != TensorProto.FLOAT:
            raise RuntimeError(f"external initializer {init.name} is not float32")
        n = int(np.prod(init.dims)) if len(init.dims) else 1
        arr = test_weights(init.name, n).reshape(list(init.dims))
        init.CopyFrom(numpy_helper.from_array(arr, init.name))
    return model


def read_meta(d):
    ins, ints, outs = {}, {}, []
    for line in open(os.path.join(d, "meta.txt")):
        t = line.split()
        if t[0] == "in":
            ins[t[1]] = (t[2], [int(v) for v in t[3:]])
        elif t[0] == "int":
            eq = t.index("=")
            ints[t[1]] = ([int(v) for v in t[2:eq]], [int(v) for v in t[eq + 1:]])
        elif t[0] == "out":
            outs.append((t[1], [int(v) for v in t[2:]]))
    return ins, ints, outs


def main():
    dump = sys.argv[1]
    only = set(sys.argv[2:])
    worst = 0
    for name in sorted(os.listdir(dump)):
        if only and name not in only:
            continue
        d = os.path.join(dump, name)
        ins, ints, outs = read_meta(d)
        feeds = {}
        for k, (kind, shape) in ins.items():
            a = np.fromfile(os.path.join(d, f"in_{k}.f32"), np.float32).reshape(shape)
            feeds[k] = a.astype(bool) if kind == "bool" else a
        for k, (shape, vals) in ints.items():
            feeds[k] = np.array(vals, np.int64).reshape(shape)
        model = load_with_weights(os.path.join(RESOURCES, f"graph_{name}.onnx"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        total = sum(int(np.prod(i.dims)) if len(i.dims) else 1 for i in model.graph.initializer)
        if total > 3e8:  # a multi-GB model: weights go to a temporary external-data file instead of one in-memory protobuf
            import tempfile
            with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as tmp:
                path = os.path.join(tmp, "m.onnx")
                onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True, location="m.weights", size_threshold=1024)
                del model
                sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
                got = sess.run([o for o, _ in outs], feeds)
                del sess
        else:
            sess = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
            got = None
        if got is None:
            got = sess.run([o for o, _ in outs], feeds)
        report = []
        for (oname, shape), want in zip(outs, got):
            mine = np.fromfile(os.path.join(d, f"out_{oname}.f32"), np.float32)
            ref = np.asarray(want).astype(np.float32).reshape(-1)
            if mine.size != ref.size:
                report.append(f"{oname}: size {mine.size} vs {ref.size}")
                continue
            nan_a, nan_b = np.isnan(mine), np.isnan(ref)
            if (nan_a != nan_b).any():
                report.append(f"{oname}: NaN pattern differs ({nan_a.sum()} vs {nan_b.sum()})")
                continue
            fin = ~nan_a & np.isfinite(mine) & np.isfinite(ref)
            diff = float(np.abs(mine[fin] - ref[fin]).max()) if fin.any() else 0.0
            scale = float(np.abs(ref[fin]).max()) if fin.any() else 0.0
            if (np.isinf(mine) != np.isinf(ref)).any():
                report.append(f"{oname}: infinities differ")
            elif diff > 1e-3 + 1e-3 * scale:
                report.append(f"{oname}: max|diff| {diff:.3g} at scale {scale:.3g}")
            worst = max(worst, diff / (1e-3 + 1e-3 * scale))
        print(("FAIL " if report else "ok   ") + name + ("  " + "; ".join(report[:3]) if report else ""), flush=True)
    return 0


sys.exit(main())
