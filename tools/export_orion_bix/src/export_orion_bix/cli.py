"""CLI: uv run export_orion_bix --config real --out DIR.

Writes into --out:
  graph_orion_bix_classification.onnx       weight-free graph (checkpoint
                                            initializers are EXTERNAL stubs; the
                                            .onnx.data is deleted)
  tensor_map_orion_bix_classification.json  ONNX initializer name -> safetensors key
  export_report_orion_bix_classification.json  provenance + parity numbers

Orion-BiX is classification-only (upstream ships no regression head), so unlike
the other exporters there is no --task flag.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from export_orion_bix import configs, export
from export_orion_bix.orion_bix_patches import build_model

TASK = "classification"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="export_orion_bix")
    ap.add_argument("--config", required=True, choices=["fixture", "real"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args(argv)

    cfg = configs.get(args.config)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / f"graph_orion_bix_{TASK}.onnx"
    map_path = out / f"tensor_map_orion_bix_{TASK}.json"

    print(f"[export_orion_bix] building random-weight OrionBix ({args.config}) ...", flush=True)
    t0 = time.time()
    model = build_model(cfg.model_kwargs, seed=args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export_orion_bix] {n_params:,} params ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    wrapper = export.export_graph(model, graph_path, dim_rows=cfg.dim_rows,
                                  dim_train=cfg.dim_train, dim_features=cfg.dim_features,
                                  example=cfg.example)
    print(f"[export_orion_bix] dynamo export done ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    tensor_map = export.postprocess(graph_path, dict(model.state_dict()))
    print(f"[export_orion_bix] postprocess done ({time.time()-t0:.1f}s): "
          f"{len(tensor_map['initializers'])} initializers mapped, "
          f"{len(tensor_map['unmatched_small'])} small inline constants", flush=True)
    export.write_tensor_map(map_path, tensor_map, task=TASK,
                            safetensors_rel=f"{TASK}/model.safetensors")

    parity = None
    if not args.skip_parity:
        t0 = time.time()
        parity = export.check_parity(graph_path, wrapper, cfg.parity_shapes)
        print(f"[export_orion_bix] parity ({time.time()-t0:.1f}s): worst {parity['worst']:.2e} "
              f"(budget {parity['tol']:.0e}) -> {'OK' if parity['ok'] else 'FAIL'}", flush=True)
        if not parity["ok"]:
            print("[export_orion_bix] PARITY FAILED", file=sys.stderr)
            return 1

    export.delete_weight_data(graph_path)
    export.assert_weight_free(graph_path, tensor_map)

    import onnx as _onnx
    import onnxruntime as _ort
    import onnxscript as _onnxscript
    import torch as _torch
    report = {
        "command": ["export_orion_bix"] + list(argv or sys.argv[1:]),
        "task": TASK, "config": cfg.name, "model_kwargs": cfg.model_kwargs,
        "n_params": n_params, "seed": args.seed, "opset": export.OPSET,
        "upstream": {
            "repo": "https://github.com/Lexsi-Labs/Orion-BiX",
            "rev": "060d34606f56c591c47e0b6cbe971e5f5035f33e",
            "license": "MIT",
            "checkpoint": "Lexsi/Orion-BiX :: Orion-BiX-v1.1.ckpt",
        },
        "input_signature": {"x": "[1,T,H] f32", "y": "[1,S] f32 (S=train_size, train labels only)"},
        "output": {"logits": "[1,T,C] class logits, C=max_classes"},
        "H_dynamic": True, "train_size": "runtime (implicit = len(y))",
        "cat_mask": "omitted (no categorical path)",
        "d": "omitted (engine pads features; d-path uses boolean-mask indexing)",
        "dim_rows": list(cfg.dim_rows), "dim_train": list(cfg.dim_train),
        "dim_features": list(cfg.dim_features), "example_THS": list(cfg.example),
        "graph_bytes": graph_path.stat().st_size,
        "n_initializers_mapped": len(tensor_map["initializers"]),
        "unmatched_small_inline": tensor_map["unmatched_small"],
        "parity": parity,
        "versions": {"torch": _torch.__version__, "onnx": _onnx.__version__,
                     "onnxruntime": _ort.__version__, "onnxscript": _onnxscript.__version__},
    }
    (out / f"export_report_orion_bix_{TASK}.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"[export_orion_bix] graph: {graph_path} "
          f"({graph_path.stat().st_size/1e6:.2f} MB, weight-free)")
    print(f"[export_orion_bix] map:   {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
