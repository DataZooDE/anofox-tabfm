#!/usr/bin/env python3
"""Standalone reproducer: CUDA EP returns a Slice untrimmed.

Fetches a public weight-free ONNX graph, injects synthesized initializers (the
graph ships without weights by design), and runs it on the CUDA EP. The graph
runs correctly on the CPU EP and fails on CUDA at a ScatterND whose `indices`
were meant to be trimmed to the length of a runtime bound.

    pip install onnxruntime-gpu==1.23.2 onnx numpy 'nvidia-cudnn-cu12<10'
    python ort_repro.py            # both providers
    python ort_repro.py --pinned   # same graph + bound tensors as graph outputs
"""
import argparse
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort

RAW = "https://raw.githubusercontent.com/DataZooDE/anofox-tabfm/{ref}/resources/graph_tabicl_classification.onnx"
PRE_FIX = "v2026.08.13"   # before the workaround
POST_FIX = "v2026.08.14"  # with the bound tensors named as graph outputs


def fetch(ref: str) -> str:
    path = f"/tmp/graph_{ref}.onnx"
    try:
        open(path, "rb").close()
    except OSError:
        urllib.request.urlretrieve(RAW.format(ref=ref), path)
    return path


def initializers(path: str, seed: int = 0):
    """The graph ships weight-free; synthesize values for its external stubs."""
    model = onnx.load(path, load_external_data=False)
    rng = np.random.default_rng(seed)
    names, values = [], []
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        dtype = {1: np.float32, 7: np.int64, 9: np.bool_}[init.data_type]
        dims = list(init.dims)
        arr = ((rng.standard_normal(dims) * 0.02).astype(dtype) if dtype == np.float32
               else rng.integers(0, 2, dims).astype(dtype))
        names.append(init.name)
        values.append(ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr)))
    return names, values


def run(path: str, provider: str, T: int, S: int, H: int = 3):
    names, values = initializers(path)
    so = ort.SessionOptions()
    so.add_external_initializers(names, values)
    so.log_severity_level = 3
    sess = ort.InferenceSession(path, so, providers=[provider])
    if provider not in sess.get_providers():
        return f"SKIP: {provider} did not instantiate (got {sess.get_providers()})"
    rng = np.random.default_rng(42)
    feeds = {"x": rng.standard_normal((1, T, H)).astype(np.float32),
             "y": rng.integers(0, 2, (1, S)).astype(np.float32)}
    try:
        out = sess.run(["logits"], feeds)[0]
        return f"OK   logits {tuple(out.shape)}"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL {str(exc).strip().splitlines()[-1][:200]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", action="store_true",
                    help="use the graph whose slice bounds are named as outputs")
    args = ap.parse_args()

    ref = POST_FIX if args.pinned else PRE_FIX
    path = fetch(ref)
    print(f"onnxruntime {ort.__version__}   graph {ref}"
          f"{'  (bounds pinned as graph outputs)' if args.pinned else ''}")
    for T, S in [(70, 60), (128, 100)]:
        print(f"  T={T} S={S}  (context={S}, total={T})")
        for provider in ("CPUExecutionProvider", "CUDAExecutionProvider"):
            print(f"    {provider:24} {run(path, provider, T, S)}")


if __name__ == "__main__":
    main()
