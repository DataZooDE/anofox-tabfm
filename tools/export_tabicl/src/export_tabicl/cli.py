"""CLI: uv run export_tabicl --task classification --config real --out DIR.

Writes into --out:
  graph_tabicl_<task>.onnx       weight-free graph (checkpoint initializers are
                                 EXTERNAL stubs; the .onnx.data is deleted)
  tensor_map_tabicl_<task>.json  ONNX initializer name -> safetensors key
  export_report_tabicl_<task>.json  provenance + parity numbers
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from export_tabicl import configs, export
from export_tabicl.tabicl_patches import build_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="export_tabicl")
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--config", required=True, choices=["fixture", "real"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args(argv)

    cfg = configs.get(args.config, task=args.task)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / f"graph_tabicl_{args.task}.onnx"
    map_path = out / f"tensor_map_tabicl_{args.task}.json"

    print(f"[export_tabicl] building random-weight TabICL ({args.config}, {args.task}) ...",
          flush=True)
    t0 = time.time()
    model = build_model(args.task, cfg.model_kwargs, seed=args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export_tabicl] {n_params:,} params ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    wrapper = export.export_graph(model, graph_path, dim_rows=cfg.dim_rows,
                                  dim_train=cfg.dim_train, dim_features=cfg.dim_features,
                                  example=cfg.example)
    print(f"[export_tabicl] dynamo export done ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    tensor_map = export.postprocess(graph_path, dict(model.state_dict()))
    print(f"[export_tabicl] postprocess done ({time.time()-t0:.1f}s): "
          f"{len(tensor_map['initializers'])} initializers mapped, "
          f"{len(tensor_map['unmatched_small'])} small inline constants", flush=True)
    export.write_tensor_map(map_path, tensor_map, task=args.task,
                            safetensors_rel=f"{args.task}/model.safetensors")

    parity = None
    if not args.skip_parity:
        t0 = time.time()
        parity = export.check_parity(graph_path, wrapper, cfg.parity_shapes)
        print(f"[export_tabicl] parity ({time.time()-t0:.1f}s): worst {parity['worst']:.2e} "
              f"(budget {parity['tol']:.0e}) -> {'OK' if parity['ok'] else 'FAIL'}", flush=True)
        if not parity["ok"]:
            print("[export_tabicl] PARITY FAILED", file=sys.stderr)
            return 1

    export.delete_weight_data(graph_path)
    export.assert_weight_free(graph_path, tensor_map)

    import onnx as _onnx
    import onnxruntime as _ort
    import onnxscript as _onnxscript
    import torch as _torch
    report = {
        "command": ["export_tabicl"] + list(argv or sys.argv[1:]),
        "task": args.task, "config": cfg.name, "model_kwargs": cfg.model_kwargs,
        "n_params": n_params, "seed": args.seed, "opset": export.OPSET,
        "input_signature": {"x": "[1,T,H] f32", "y": "[1,S] f32 (S=train_size, train labels only)"},
        "output": (
            {"logits": "[1,T,C] class logits, C=max_classes"} if args.task == "classification"
            else {"logits": "[1,T,1] single real-valued point estimate per row "
                            "(mean over the 999-quantile head + in-graph inverse "
                            "StandardScaler); engine feeds RAW train targets and reads "
                            "the output directly (no target z-score, no inverse-transform)"}),
        "H_dynamic": True, "train_size": "runtime (implicit = len(y))",
        "cat_mask": "omitted (no categorical path)", "d": "omitted (unsupported with feature grouping)",
        "dim_rows": list(cfg.dim_rows), "dim_train": list(cfg.dim_train),
        "dim_features": list(cfg.dim_features), "example_THS": list(cfg.example),
        "graph_bytes": graph_path.stat().st_size,
        "n_initializers_mapped": len(tensor_map["initializers"]),
        "unmatched_small_inline": tensor_map["unmatched_small"],
        "parity": parity,
        "versions": {"torch": _torch.__version__, "onnx": _onnx.__version__,
                     "onnxruntime": _ort.__version__, "onnxscript": _onnxscript.__version__},
    }
    (out / f"export_report_tabicl_{args.task}.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"[export_tabicl] graph: {graph_path} "
          f"({graph_path.stat().st_size/1e6:.2f} MB, weight-free)")
    print(f"[export_tabicl] map:   {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
