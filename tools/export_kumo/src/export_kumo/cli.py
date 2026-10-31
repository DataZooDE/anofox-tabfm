"""CLI: uv run export_kumo --task classification --size small --out DIR.

Writes into --out:
  graph_kumo_<size>_<task>.onnx          weight-free graph (checkpoint initializers are EXTERNAL
                                         stubs; the .onnx.data is deleted)
  tensor_map_kumo_<size>_<task>.json     ONNX initializer name -> checkpoint state_dict key
  export_report_kumo_<size>_<task>.json  provenance + parity numbers

`--size fixture` is the tiny random-init CI model (test/fixtures/kumo); the other three are the
published sizes, with the dims of upstream's own MODEL_KWARGS. Weights are random-init either way:
the graph is weight-free and the tensor map is a property of the module tree, not of values.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from export_kumo import configs, export
from export_kumo.kumo_patches import build_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="export_kumo")
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--size", required=True, choices=["fixture", "small", "medium", "large"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args(argv)

    cfg = configs.get(args.size)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"kumo_{args.size}_{args.task}"
    graph_path = out / f"graph_{stem}.onnx"
    map_path = out / f"tensor_map_{stem}.json"
    regression = args.task == "regression"
    classes = 0 if regression else cfg.classes

    print(f"[export_kumo] building random-weight Kumo-Tabular ({args.size}, {args.task}) ...", flush=True)
    t0 = time.time()
    model = build_model(args.task, cfg.model, seed=args.seed, num_classes=cfg.classes,
                        num_quantiles=cfg.quantiles)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export_kumo] {n_params:,} params ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    wrapper = export.export_graph(model, graph_path, example=cfg.example, classes=classes,
                                  regression=regression)
    print(f"[export_kumo] dynamo export done ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    tensor_map = export.postprocess(graph_path, dict(model.state_dict()))
    print(f"[export_kumo] postprocess done ({time.time()-t0:.1f}s): "
          f"{len(tensor_map['initializers'])} initializers mapped, "
          f"{len(tensor_map['unmatched_small'])} small inline constants", flush=True)
    export.write_tensor_map(map_path, tensor_map, task=args.task,
                            safetensors_rel=f"{args.size}/{args.task}/model.safetensors")

    parity = None
    if not args.skip_parity:
        t0 = time.time()
        parity = export.check_parity(graph_path, wrapper, cfg.parity_shapes, classes)
        print(f"[export_kumo] parity ({time.time()-t0:.1f}s): worst {parity['worst']:.2e} "
              f"(budget {parity['tol']:.0e}, every row) -> {'OK' if parity['ok'] else 'FAIL'}", flush=True)
        if not parity["ok"]:
            print("[export_kumo] PARITY FAILED", file=sys.stderr)
            return 1

    export.delete_weight_data(graph_path)
    export.assert_weight_free(graph_path, tensor_map)

    import onnx as _onnx
    import onnxruntime as _ort
    import onnxscript as _onnxscript
    import torch as _torch
    report = {
        "command": ["export_kumo"] + list(argv or sys.argv[1:]),
        "task": args.task, "size": args.size, "model": cfg.model,
        "outputs": cfg.classes if not regression else cfg.quantiles,
        "n_params": n_params, "seed": args.seed, "opset": export.OPSET,
        "upstream": "NVIDIA/structured-data-models @ 084aa7c7d69403d453c0073b785f1d32915669a8, "
                    "sdm.models.kumo.tabular._KumoTabular.forward_train",
        "input_signature": {
            "x": "[1,T,H] f32: engine-standardised features (kumo_tabular_v1_minimal)",
            "y": "[1,T] f32: context labels in rows < train_size (class ids, or the engine-"
                 "standardised regression target); the -100 sentinel elsewhere, masked in-graph",
            "train_size": "[1] i64: context rows (a value, not a shape)",
            "cat_mask": "[1,H] bool: categorical columns",
            "d": "[1] i64: real feature columns (<= H; the rest is bucket padding)",
        },
        "output": (
            {"logits": "[1,T,10] class logits; rows < train_size are in-context fitted values"}
            if not regression else
            {"logits": "[1,T,1] mean of the 999 quantiles in the standardised target space; "
                       "the engine inverts its own target z-score"}),
        "split": "masked: arange(T) < train_size; padded columns masked by d",
        "graph_bytes": graph_path.stat().st_size,
        "n_initializers_mapped": len(tensor_map["initializers"]),
        "unmatched_small_inline": tensor_map["unmatched_small"],
        "parity": parity,
        "versions": {"torch": _torch.__version__, "onnx": _onnx.__version__,
                     "onnxruntime": _ort.__version__, "onnxscript": _onnxscript.__version__},
    }
    (out / f"export_report_{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[export_kumo] graph: {graph_path} ({graph_path.stat().st_size/1e6:.2f} MB, weight-free)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
