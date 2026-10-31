"""CLI: uv run export_nori --config real --out DIR.

Writes into --out (Nori is regression-only, so there is one task):
  graph_nori_regression.onnx           weight-free graph (checkpoint initializers are EXTERNAL stubs;
                                       the .onnx.data is deleted)
  tensor_map_nori_regression.json      ONNX initializer name -> checkpoint key
  export_report_nori_regression.json   provenance + parity numbers

The shipped copies live in resources/ as graph_nori30m_regression.onnx etc. (renamed on copy, so a
future Nori size can never be mistaken for this one).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from export_nori import configs, export
from export_nori.nori_patches import build_model

TASK = "regression"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="export_nori")
    ap.add_argument("--config", required=True, choices=["fixture", "real"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--contract", choices=["positional", "mask"], default="positional",
                    help="positional: (x, y) with the split = len(y) (the CPU graph). mask: x, y (full length), "
                         "train_size, d, n_rows -- the ROCm/MIGraphX contract (nori_mask.py).")
    args = ap.parse_args(argv)
    masked = args.contract == "mask"

    cfg = configs.get(args.config)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / (f"graph_mask_nori_{TASK}.onnx" if masked else f"graph_nori_{TASK}.onnx")
    map_path = out / (f"tensor_map_mask_nori_{TASK}.json" if masked else f"tensor_map_nori_{TASK}.json")

    print(f"[export_nori] building random-weight Nori ({args.config}) ...", flush=True)
    t0 = time.time()
    model = build_model(cfg.model_config, seed=args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export_nori] {n_params:,} params ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    if masked:
        wrapper = export.export_mask_graph(model, graph_path, dim_rows=cfg.dim_rows,
                                           dim_features=cfg.dim_features, example=cfg.example)
    else:
        wrapper = export.export_graph(model, graph_path, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                                      dim_features=cfg.dim_features, example=cfg.example)
    print(f"[export_nori] dynamo export done ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    tensor_map = export.postprocess(graph_path, dict(model.state_dict()))
    print(f"[export_nori] postprocess done ({time.time()-t0:.1f}s): "
          f"{len(tensor_map['initializers'])} initializers mapped, "
          f"{len(tensor_map['unmatched_small'])} small inline constants", flush=True)
    export.write_tensor_map(map_path, tensor_map, task=TASK, safetensors_rel="model.ckpt")

    parity = None
    if masked and not args.skip_parity:
        from export_nori.mask_parity import check_onnx
        t0 = time.time()
        parity = check_onnx(graph_path, model)
        print(f"[export_nori] masked ORT-vs-positional parity ({time.time()-t0:.1f}s): worst "
              f"{parity['worst']:.2e} -> {'OK' if parity['ok'] else 'FAIL'}", flush=True)
        if not parity["ok"]:
            print("[export_nori] PARITY FAILED", file=sys.stderr)
            return 1
    elif not args.skip_parity:
        t0 = time.time()
        parity = export.check_parity(graph_path, wrapper, cfg.parity_shapes)
        print(f"[export_nori] parity ({time.time()-t0:.1f}s): worst {parity['worst']:.2e} "
              f"(budget {parity['tol']:.0e}) -> {'OK' if parity['ok'] else 'FAIL'}", flush=True)
        if not parity["ok"]:
            print("[export_nori] PARITY FAILED", file=sys.stderr)
            return 1

    export.delete_weight_data(graph_path)
    export.assert_weight_free(graph_path, tensor_map)

    from importlib.metadata import version

    import onnx as _onnx
    import onnxruntime as _ort
    import onnxscript as _onnxscript
    import torch as _torch
    report = {
        "command": ["export_nori"] + list(argv or sys.argv[1:]),
        "task": TASK, "config": cfg.name, "model_config": cfg.model_config,
        "n_params": n_params, "seed": args.seed, "opset": export.OPSET,
        "upstream": {
            "package": f"synthefy-nori=={version('synthefy-nori')}",
            "repo": "https://github.com/Synthefy/synthefy-nori",
            "code_license": "Apache-2.0",
            "weights_license": "Apache-2.0",
            "checkpoint": "Synthefy/Nori-30M@63c9f7facf9fb32c37ce3fc2fba331d524696318 :: nori.pt",
        },
        "input_signature": {"x": "[1,T,H] f32", "y": "[1,S] f32 (S=eval_pos, RAW training targets only)"},
        "output": {"logits": "[1,T,1] RAW-space point estimate per row: the mean of the 999-quantile "
                             "head (upstream's default quantile_collapse='mean')"},
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
    if masked:
        report["input_signature"] = {"x": "[1,T,H] f32", "y": "[1,T] f32 FULL length (-100 off-context)",
                                     "train_size": "[1] i64", "d": "[1] i64", "n_rows": "[1] i64"}
    (out / f"export_report_{'mask_' if masked else ''}nori_{TASK}.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"[export_nori] graph: {graph_path} ({graph_path.stat().st_size/1e6:.2f} MB, weight-free)")
    print(f"[export_nori] map:   {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
