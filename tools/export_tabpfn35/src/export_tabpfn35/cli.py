"""CLI: uv run export_tabpfn35 --task classification --config real35 --out DIR.

Writes into --out (stems: tabpfn35 for TabPFN-3.5, tabpfn35fast for 3.5-Fast):
  graph_<stem>_<task>.onnx          positional contract, weight-free
  graph_mask_<stem>_<task>.onnx     masked contract (--contract mask), weight-free;
                                    tools/make_migraphx_graph.py turns it into the
                                    bundled graph_migraphx_<stem>_<task>.onnx
  tensor_map_<stem>_<task>.json     ONNX initializer -> safetensors key
  export_report_<stem>_<task>.json  provenance + parity numbers

Parity, always on EVERY row: ORT on the exported graph vs the PyTorch wrapper,
and the PyTorch wrapper vs upstream's own TabPFNV3p5.forward.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

from export_tabpfn import export as xp
from export_tabpfn35 import configs
from export_tabpfn35.model import build_random_model
from export_tabpfn35.wrappers import MaskWrapper35, Wrapper35, upstream_wrapped

OPSET = 18


def _example(t, h, n, task, masked):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, t, h, generator=g)
    rows = t if masked else n
    if task == "regression":
        y = torch.randn(1, rows, generator=g)
    else:
        y = torch.arange(rows, dtype=torch.float32).remainder(3)[None]
    if not masked:
        return (x, y)
    i = lambda v: torch.tensor([v], dtype=torch.int64)  # noqa: E731
    return (x, y, i(n), i(t), i(h))


def export_graph(model, path, *, task, example, masked):
    wrapper = (MaskWrapper35 if masked else Wrapper35)(model, task=task).eval()
    A, S = torch.export.Dim.AUTO, torch.export.Dim.STATIC
    if masked:
        dyn = ({0: S, 1: A, 2: A}, {0: S, 1: A}, None, None, None)
        names = ["x", "y", "train_size", "n_rows", "d"]
    else:
        dyn = ({0: S, 1: A, 2: A}, {0: S, 1: A})
        names = ["x", "y"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrapper, _example(*example, task, masked), str(path),
                          dynamo=True, dynamic_shapes=dyn, opset_version=OPSET,
                          input_names=names, output_names=["logits"],
                          external_data=True, optimize=False)
    return wrapper


def _feed(t, h, n, task, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, t, h)).astype(np.float32)
    x[:, :, 0] = rng.integers(0, 3, (1, t))                 # a tied column
    if task == "regression":
        y = (rng.standard_normal((1, n)) * 3.0 + 7.0).astype(np.float32)
    else:
        y = rng.integers(0, 3, (1, n)).astype(np.float32)
    return x, y


def write_map(path, tensor_map, *, task, safetensors_rel):
    """tools/export_tabpfn's tensor map, plus `"complete": true`.

    3.5 ships ONE weights file per variant holding both heads, so the file has
    tensors the other task's graph does not declare. The flag tells the engine the
    map names every initializer of THIS graph, so it injects exactly those and
    skips the rest instead of offering them to ORT under the `m.` fallback name
    (src/tabfm_engine.cpp TensorMapDeclaresComplete). Only true because
    postprocess() maps every external initializer -- asserted here.
    """
    import json

    if tensor_map["unmatched_small"]:
        raise RuntimeError(f"cannot mark {path.name} complete: unmapped initializers "
                           f"{tensor_map['unmatched_small'][:5]}")
    xp.write_tensor_map(path, tensor_map, task=task, safetensors_rel=safetensors_rel)
    doc = json.loads(path.read_text())
    doc["complete"] = True
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def check_parity(path, model, shapes, *, task, masked, tol=1e-3):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    wrap = Wrapper35(model, task=task).eval()
    out, worst_ort, worst_up = [], 0.0, 0.0
    for t, h, n in shapes:
        x, y = _feed(t, h, n, task)
        if masked:
            # pad to a larger bucket, the way the plugin does
            Tp, Hp = t + 9, h + 3
            xp_ = np.zeros((1, Tp, Hp), np.float32)
            xp_[:, :t, :h] = x
            yp = np.full((1, Tp), -100.0, np.float32)
            yp[:, :n] = y
            i = lambda v: np.array([v], np.int64)  # noqa: E731
            feed = {"x": xp_, "y": yp, "train_size": i(n), "n_rows": i(t), "d": i(h)}
        else:
            feed = {"x": x, "y": y}
        t0 = time.time()
        (o,) = sess.run(["logits"], feed)
        ms = (time.time() - t0) * 1e3
        o = o[:, :t]
        with torch.no_grad():
            pt = wrap(torch.from_numpy(x), torch.from_numpy(y)).numpy()
            up = upstream_wrapped(model, torch.from_numpy(x), torch.from_numpy(y), task).numpy()
        d_ort = float(np.abs(o - pt).max())
        d_up = float(np.abs(pt - up).max())
        argmax = True if task == "regression" else bool((o.argmax(-1) == up.argmax(-1)).all())
        worst_ort, worst_up = max(worst_ort, d_ort), max(worst_up, d_up)
        out.append({"T": t, "H": h, "train": n, "ort_vs_torch": d_ort,
                    "torch_vs_upstream": d_up, "argmax_agree_upstream": argmax, "ort_ms": ms})
    ok = worst_ort < tol and worst_up < tol and all(r["argmax_agree_upstream"] for r in out)
    return {"ok": ok, "worst_ort_vs_torch": worst_ort, "worst_torch_vs_upstream": worst_up,
            "tol": tol, "shapes": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="export_tabpfn35")
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--config", required=True, choices=sorted(configs.CONFIGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contract", choices=["positional", "mask"], default="positional")
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--safetensors-rel", default=None,
                    help="tensor-map `safetensors` field (default: <task>/model.safetensors)")
    args = ap.parse_args(argv)
    masked = args.contract == "mask"
    cfg = configs.get(args.config)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = cfg.slug
    graph_path = out / (f"graph_mask_{stem}_{args.task}.onnx" if masked
                        else f"graph_{stem}_{args.task}.onnx")
    map_path = out / f"tensor_map_{stem}_{args.task}.json"

    t0 = time.time()
    model = build_random_model(cfg.model_kwargs, seed=args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export_tabpfn35] {cfg.name} {args.task}: {n_params:,} params "
          f"({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    export_graph(model, graph_path, task=args.task, example=cfg.example, masked=masked)
    print(f"[export_tabpfn35] dynamo export ({time.time()-t0:.1f}s)", flush=True)

    sd = dict(model.state_dict())
    tensor_map = xp.postprocess(graph_path, sd)
    print(f"[export_tabpfn35] {len(tensor_map['initializers'])} initializers mapped, "
          f"{len(tensor_map['unmatched_small'])} small inline constants "
          f"{tensor_map['unmatched_small']}", flush=True)
    write_map(map_path, tensor_map, task=args.task,
              safetensors_rel=args.safetensors_rel or f"{args.task}/model.safetensors")

    parity = None
    if not args.skip_parity:
        t0 = time.time()
        parity = check_parity(graph_path, model, cfg.parity_shapes, task=args.task, masked=masked)
        print(f"[export_tabpfn35] parity ({time.time()-t0:.1f}s): ORT-vs-torch "
              f"{parity['worst_ort_vs_torch']:.2e}, torch-vs-upstream "
              f"{parity['worst_torch_vs_upstream']:.2e} -> {'OK' if parity['ok'] else 'FAIL'}",
              flush=True)
        if not parity["ok"]:
            print(json.dumps(parity, indent=2), file=sys.stderr)
            return 1

    xp.delete_weight_data(graph_path)
    xp.assert_weight_free(graph_path, tensor_map)

    import onnx as _onnx
    import onnxruntime as _ort
    import tabpfn as _tabpfn
    report = {
        "command": ["export_tabpfn35"] + list(argv or sys.argv[1:]),
        "task": args.task, "config": cfg.name, "arch": "v3.5", "contract": args.contract,
        "model_kwargs": cfg.model_kwargs, "n_params": n_params, "seed": args.seed,
        "opset": OPSET, "example_input_THN": list(cfg.example),
        "graph_bytes": graph_path.stat().st_size,
        "n_initializers_mapped": len(tensor_map["initializers"]),
        "unmatched_small_inline": tensor_map["unmatched_small"],
        "parity": parity,
        "versions": {"torch": torch.__version__, "onnx": _onnx.__version__,
                     "onnxruntime": _ort.__version__,
                     "tabpfn": getattr(_tabpfn, "__version__", "9.2.0")},
    }
    if not masked:
        (out / f"export_report_{stem}_{args.task}.json").write_text(
            json.dumps(report, indent=2) + "\n")
    print(f"[export_tabpfn35] graph: {graph_path} ({graph_path.stat().st_size/1e6:.2f} MB, "
          "weight-free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def injected_session(graph_path, initializers: dict, tensors):
    """ORT session on a weight-free graph with `tensors[key]` injected in memory.

    `initializers` is the tensor map's {onnx name -> safetensors key}. The proto is
    patched and loaded from bytes, because ORT validates external-data paths at
    load time, before any SessionOptions.add_initializer override applies.
    """
    import onnx
    import onnxruntime as ort

    proto = onnx.load(str(graph_path), load_external_data=False)
    by_name = {i.name: i for i in proto.graph.initializer}
    for name, key in initializers.items():
        t = tensors[key]
        arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        by_name[name].CopyFrom(onnx.numpy_helper.from_array(np.ascontiguousarray(arr), name))
    return ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
