"""CLI: uv run export_limix --task classification --config real --out DIR.

Writes into --out:
  graph_limix_<task>.onnx           weight-free graph (checkpoint initializers
                                    are EXTERNAL stubs; the .onnx.data is deleted)
  tensor_map_limix_<task>.json      ONNX initializer name -> safetensors key
  export_report_limix_<task>.json   provenance + parity numbers
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from export_limix import configs, export
from export_limix.limix_patches import VENDOR, build_model


def _vendor_rev() -> str:
    head = VENDOR.parent.parent / ".git"
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="export_limix")
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--config", required=True, choices=["fixture", "real"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--contract", choices=["positional", "mask"], default="positional",
                    help="positional: (x, y) with the split = len(y) (the CPU graph). mask: x, y (full length), "
                         "train_size, d, n_rows -- the ROCm/MIGraphX contract (limix_mask.py).")
    args = ap.parse_args(argv)
    masked = args.contract == "mask"

    cfg = configs.get(args.config)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"mask_limix_{args.task}" if masked else f"limix_{args.task}"
    graph_path = out / f"graph_{stem}.onnx"
    map_path = out / f"tensor_map_{stem}.json"

    print(f"[export_limix] building random-weight LimiX ({args.config}, {args.task}) ...",
          flush=True)
    t0 = time.time()
    model = build_model(cfg.model_config, seed=args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export_limix] {n_params:,} params ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    if masked:
        wrapper = export.export_mask_graph(model, graph_path, task=args.task, dim_rows=cfg.dim_rows,
                                           dim_features=cfg.dim_features, example=cfg.example,
                                           max_classes=cfg.max_classes)
    else:
        wrapper = export.export_graph(model, graph_path, dim_rows=cfg.dim_rows,
                                      dim_train=cfg.dim_train, dim_features=cfg.dim_features,
                                      example=cfg.example, max_classes=cfg.max_classes,
                                      task=args.task)
    print(f"[export_limix] dynamo export done ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    tensor_map = export.postprocess(graph_path, dict(model.state_dict()))
    print(f"[export_limix] postprocess done ({time.time()-t0:.1f}s): "
          f"{len(tensor_map['initializers'])} initializers mapped, "
          f"{len(tensor_map['unmatched_small'])} small inline constants", flush=True)
    export.write_tensor_map(map_path, tensor_map, task=args.task,
                            safetensors_rel=f"{args.task}/model.safetensors")

    parity = None
    if masked and not args.skip_parity:
        from export_limix.mask_parity import check_onnx
        t0 = time.time()
        parity = check_onnx(graph_path, model, args.task)
        print(f"[export_limix] masked ORT-vs-positional parity ({time.time()-t0:.1f}s): worst "
              f"{parity['worst']:.2e} -> {'OK' if parity['ok'] else 'FAIL'}", flush=True)
        if not parity["ok"]:
            print("[export_limix] PARITY FAILED", file=sys.stderr)
            return 1
    elif not args.skip_parity:
        t0 = time.time()
        parity = export.check_parity(graph_path, wrapper, cfg.parity_shapes,
                                     max_classes=cfg.max_classes, task=args.task)
        print(f"[export_limix] parity ({time.time()-t0:.1f}s): worst {parity['worst']:.2e} "
              f"(budget {parity['tol']:.0e}) -> {'OK' if parity['ok'] else 'FAIL'}", flush=True)
        if not parity["ok"]:
            print("[export_limix] PARITY FAILED", file=sys.stderr)
            return 1

    export.delete_weight_data(graph_path)
    export.assert_weight_free(graph_path, tensor_map)

    import onnx as _onnx
    import onnxruntime as _ort
    import onnxscript as _onnxscript
    import torch as _torch
    report = {
        "command": ["export_limix"] + list(argv or sys.argv[1:]),
        "task": args.task, "config": cfg.name, "model_config": cfg.model_config,
        "n_params": n_params, "seed": args.seed, "opset": export.OPSET,
        "upstream": {
            "repo": "https://github.com/limix-ldm-ai/LimiX",
            "rev": _vendor_rev(),
            "code_license": "Apache-2.0",
            "weights_license": "Stable AI Technology Co., Ltd. License v1.0 (Apache-2.0 + Section 10: "
                               "'Built with StableAI LimiX' attribution; 'LimiX' name prefix for derived models)",
            "checkpoint": "stable-ai/LimiX-2M :: LimiX-2M.ckpt",
        },
        "input_signature": {"x": "[1,T,H] f32",
                            "y": "[1,S] f32 (S=eval_pos, train labels/targets only)"},
        "output": ({"logits": "[1,T,C] class logits, C=num_classes"}
                   if args.task == "classification"
                   else {"logits": "[1,T,1] point estimate per row"}),
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
        report["input_signature"] = {"x": "[1,T,H] f32 (padded)", "y": "[1,T] f32 FULL length (-100 off-context)",
                                     "train_size": "[1] i64", "d": "[1] i64 (real feature count)",
                                     "n_rows": "[1] i64 (real row count: the valid-feature test)"}
        report["train_size"] = "input (a value: the context is a mask, not a slice)"
    (out / f"export_report_{stem}.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"[export_limix] graph: {graph_path} "
          f"({graph_path.stat().st_size/1e6:.2f} MB, weight-free)")
    print(f"[export_limix] map:   {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
