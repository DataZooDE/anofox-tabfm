"""CLI: uv run export_tabpfn --task classification --config fixture --out DIR.

Writes into --out:
  graph_tabpfn_<task>.onnx        weight-free graph (checkpoint initializers are
                                  EXTERNAL stubs; the .onnx.data is deleted)
  tensor_map_tabpfn_<task>.json   ONNX initializer name -> safetensors key
  export_report_tabpfn_<task>.json  provenance + parity numbers
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from export_tabpfn import configs, export
from export_tabpfn.tabpfn_patched import build_random_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="export_tabpfn")
    ap.add_argument("--task", required=True,
                    choices=["classification", "regression"])
    ap.add_argument("--config", required=True, choices=["tiny", "fixture", "real"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args(argv)

    cfg = configs.get(args.config)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / f"graph_tabpfn_{args.task}.onnx"
    map_path = out / f"tensor_map_tabpfn_{args.task}.json"
    n_out = cfg.max_classes if args.task == "classification" else cfg.num_buckets

    print(f"[export_tabpfn] building random-weight TabPFN v2 "
          f"({args.config}, {args.task}) ...", flush=True)
    t0 = time.time()
    kw = dict(cfg.model_kwargs)
    kw["num_buckets"] = cfg.num_buckets
    kw["max_num_classes"] = cfg.max_classes
    model = build_random_model(args.task, kw, seed=args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export_tabpfn] {n_params:,} params, n_out={model.n_out} "
          f"({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    export.export_graph(model, graph_path, example=cfg.example,
                        max_classes=cfg.max_classes)
    print(f"[export_tabpfn] dynamo export done ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    tensor_map = export.postprocess(graph_path, dict(model.state_dict()))
    print(f"[export_tabpfn] postprocess ({time.time()-t0:.1f}s): "
          f"{len(tensor_map['initializers'])} initializers mapped, "
          f"{len(tensor_map['unmatched_small'])} small inline constants",
          flush=True)
    export.write_tensor_map(map_path, tensor_map, task=args.task,
                            safetensors_rel=f"{args.task}/model.safetensors")

    parity = None
    if not args.skip_parity:
        t0 = time.time()
        parity = export.check_parity(graph_path, model, cfg.parity_shapes,
                                     max_classes=cfg.max_classes)
        print(f"[export_tabpfn] parity ({time.time()-t0:.1f}s): worst "
              f"{parity['worst']:.2e} (budget {parity['tol']:.0e}) argmax_ok "
              f"{parity['argmax_all_agree']} -> "
              f"{'OK' if parity['ok'] else 'FAIL'}", flush=True)
        if not parity["ok"]:
            print("[export_tabpfn] PARITY FAILED", file=sys.stderr)
            return 1

    export.delete_weight_data(graph_path)
    export.assert_weight_free(graph_path, tensor_map)

    import onnx as _onnx
    import onnxruntime as _ort
    import torch as _torch
    report = {
        "command": ["export_tabpfn"] + list(argv or sys.argv[1:]),
        "task": args.task, "config": cfg.name, "model_kwargs": cfg.model_kwargs,
        "n_out": model.n_out, "n_params": n_params, "seed": args.seed,
        "opset": export.OPSET, "example_input_THN": list(cfg.example),
        "graph_bytes": graph_path.stat().st_size,
        "n_initializers_mapped": len(tensor_map["initializers"]),
        "unmatched_small_inline": tensor_map["unmatched_small"],
        "parity": parity,
        "input_signature": {"x": "f32[1,T,H]", "y": "f32[1,N] (N=train_size)"},
        "output_signature": {"logits": "f32[1,T,C]"},
        "versions": {"torch": _torch.__version__, "onnx": _onnx.__version__,
                     "onnxruntime": _ort.__version__},
    }
    (out / f"export_report_tabpfn_{args.task}.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"[export_tabpfn] graph: {graph_path} "
          f"({graph_path.stat().st_size/1e6:.2f} MB, weight-free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
