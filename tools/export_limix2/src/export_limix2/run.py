"""Spike driver: python -m export_limix2.run {ref|export|parity} --layers N --task cls|reg"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

CASES = [  # (T, H, S): first is the trace shape; the rest are NOT, on purpose (baked-length guard)
    (20, 6, 12), (33, 9, 21), (50, 5, 30), (17, 11, 8), (61, 30, 40), (13, 1, 12), (40, 7, 2),
]
OUT = pathlib.Path.home() / ".cache" / "limix2-spike" / "work"


def cmd_ref(a):
    from .ref import _ref_model, make_case, reference
    m = _ref_model(a.layers)
    res = {}
    for i, (T, H, S) in enumerate(CASES):
        x, y = make_case(T, H, S, a.task, seed=100 + i)
        t = time.time(); q, al = reference(m, x, y, a.task)
        res[f"q{i}"], res[f"a{i}"] = q, al
        print(f"ref case {i} T={T} H={H} S={S}: {time.time()-t:.1f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"ref_{a.task}_{a.layers}.npz", **res)


def build_wrapper(a):
    from .limix2_patches import apply_export_patches, load_upstream
    from .wrapper import ExportWrapper2
    m, _ = load_upstream(a.layers)
    apply_export_patches(m)
    return ExportWrapper2(m, a.task).eval()


def cmd_export(a):
    from .ref import make_case
    w = build_wrapper(a)
    T, H, S = CASES[0]
    x, y = make_case(T, H, S, a.task, seed=100)
    A = torch.export.Dim.AUTO
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"limix2_{a.task}_{a.layers}.onnx"
    t = time.time()
    with torch.no_grad():
        torch.onnx.export(w, (torch.from_numpy(x), torch.from_numpy(y)), str(path), dynamo=True,
                          dynamic_shapes=({1: A, 2: A}, {1: A}), opset_version=18, input_names=["x", "y"],
                          output_names=["out"], external_data=True, optimize=False)
    print(f"onnx export OK in {time.time()-t:.0f}s -> {path} ({path.stat().st_size/1e6:.1f} MB graph)")


def cmd_parity(a):
    import onnxruntime as ort
    from .ref import make_case
    path = OUT / f"limix2_{a.task}_{a.layers}.onnx"
    ref = np.load(OUT / f"ref_{a.task}_{a.layers}.npz")
    w = build_wrapper(a)
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    worst = {}
    for i, (T, H, S) in enumerate(CASES):
        x, y = make_case(T, H, S, a.task, seed=100 + i)
        o = sess.run(None, {"x": x, "y": y})[0][0]
        with torch.inference_mode():
            e = w(torch.from_numpy(x), torch.from_numpy(y))[0].numpy()
        scale = max(1.0, float(np.abs(ref[f"a{i}"]).max()))
        d_ctx = np.abs(o[:S] - ref[f"a{i}"][:S]).max() / scale
        d_qry = np.abs(o[S:] - ref[f"q{i}"]).max() / scale
        d_all = np.abs(o - ref[f"a{i}"]).max() / scale
        d_eag = np.abs(e - ref[f"a{i}"]).max() / scale
        d_ort = np.abs(o - e).max() / scale
        print(f"case {i} T={T:3d} H={H:2d} S={S:2d} scale={scale:8.3f} | ORT vs upstream: ctx {d_ctx:.2e} qry {d_qry:.2e} "
              f"| wrapper-eager vs upstream {d_eag:.2e} | ORT vs wrapper-eager {d_ort:.2e}", flush=True)
        worst[i] = max(d_ctx, d_qry)
    print("WORST relative (ORT vs unmodified upstream, all rows):", f"{max(worst.values()):.2e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["ref", "export", "parity"])
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    {"ref": cmd_ref, "export": cmd_export, "parity": cmd_parity}[a.cmd](a)
