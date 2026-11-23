"""CLI: uv run export_exaone --task classification|regression --config real|fixture --out DIR.

Writes into --out:
  graph_exaone_<task>.onnx           weight-free graph (checkpoint initializers are EXTERNAL stubs; the
                                     .onnx.data is deleted)
  tensor_map_exaone_<task>.json      ONNX initializer name -> safetensors key
  export_report_exaone_<task>.json   provenance + parity numbers

The shipped copies live in resources/ as graph_exaone_tabular_<task>.onnx etc.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from export_exaone import configs, export
from export_exaone.exaone_patches import build_model

UPSTREAM_REV = "d149d0f0441053f0cf6b2b45f4b2067a81619cf9"
WEIGHTS = {
    "classification": "exaone-tabular-classifier-v1_default.safetensors",
    "regression": "exaone-tabular-regressor-v1_default.safetensors",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="export_exaone")
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--config", required=True, choices=["fixture", "real"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args(argv)

    cfg = configs.get(args.config, args.task)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / f"graph_exaone_{args.task}.onnx"
    map_path = out / f"tensor_map_exaone_{args.task}.json"

    print(f"[export_exaone] building random-weight EXAONE Tabular ({args.config}, {args.task}) ...", flush=True)
    t0 = time.time()
    model = build_model(args.task, cfg.model_config, cfg.regression_config, seed=args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export_exaone] {n_params:,} params ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    wrapper = export.export_graph(model, graph_path, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                                  dim_features=cfg.dim_features, example=cfg.example, task=args.task)
    print(f"[export_exaone] dynamo export done ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    tensor_map = export.postprocess(graph_path, dict(model.state_dict()))
    print(f"[export_exaone] postprocess done ({time.time()-t0:.1f}s): "
          f"{len(tensor_map['initializers'])} initializers mapped, "
          f"{len(tensor_map['unmatched_small'])} small inline constants", flush=True)
    export.write_tensor_map(map_path, tensor_map, task=args.task,
                            safetensors_rel=f"{args.task}/model.safetensors")

    parity = None
    if not args.skip_parity:
        t0 = time.time()
        parity = export.check_parity(graph_path, wrapper, cfg.parity_shapes, task=args.task)
        print(f"[export_exaone] parity ({time.time()-t0:.1f}s): worst {parity['worst']:.2e} "
              f"(budget {parity['tol']:.0e}) -> {'OK' if parity['ok'] else 'FAIL'}", flush=True)
        if not parity["ok"]:
            print("[export_exaone] PARITY FAILED", file=sys.stderr)
            return 1

    export.delete_weight_data(graph_path)
    export.assert_weight_free(graph_path, tensor_map)

    import onnx as _onnx
    import onnxruntime as _ort
    import onnxscript as _onnxscript
    import torch as _torch
    report = {
        "command": ["export_exaone"] + list(argv or sys.argv[1:]),
        "task": args.task, "config": cfg.name, "model_config": cfg.model_config,
        "regression_config": cfg.regression_config,
        "n_params": n_params, "seed": args.seed, "opset": export.OPSET,
        "upstream": {
            "package": f"exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git@{UPSTREAM_REV}",
            "code_license": "BSD-3-Clause-LG AI Research License",
            "weights_license": "EXAONE AI Model License Agreement 1.2 - NC (non-commercial)",
            "checkpoint": f"LG-AI-Research/EXAONE-Tabular :: {WEIGHTS[args.task]}",
        },
        "input_signature": {"x": "[1,T,H] f32",
                            "y": "[1,S] f32 (S=len(y), train labels / RAW targets only)"},
        "output": ({"logits": "[1,T,C] class logits, C=class_capacity"} if args.task == "classification"
                   else {"logits": "[1,T,1] RAW-space point estimate: mean of the 999-quantile head"}),
        "H_dynamic": True, "train_size": "runtime (implicit = len(y))",
        "dim_rows": list(cfg.dim_rows), "dim_train": list(cfg.dim_train),
        "dim_features": list(cfg.dim_features), "example_THS": list(cfg.example),
        "graph_bytes": graph_path.stat().st_size,
        "n_initializers_mapped": len(tensor_map["initializers"]),
        "unmatched_small_inline": tensor_map["unmatched_small"],
        "parity": parity,
        "versions": {"torch": _torch.__version__, "onnx": _onnx.__version__,
                     "onnxruntime": _ort.__version__, "onnxscript": _onnxscript.__version__},
    }
    (out / f"export_report_exaone_{args.task}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[export_exaone] graph: {graph_path} ({graph_path.stat().st_size/1e6:.2f} MB, weight-free)")
    print(f"[export_exaone] map:   {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
