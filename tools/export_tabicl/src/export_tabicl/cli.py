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

from export_tabicl import configs, export, split
from export_tabicl.tabicl_patches import build_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="export_tabicl")
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--config", required=True, choices=["fixture", "real"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--split-context", action="store_true",
                    help="ALSO export the support/query split pair (graph_prepare_*.onnx, "
                         "graph_query_*.onnx), which lets a caller encode the labelled context "
                         "once and reuse it across query batches. See split.py.")
    ap.add_argument("--contract", choices=["positional", "mask"], default="positional",
                    help="positional: (x, y) with the split = len(y) (the CPU graph). mask: x, y (full length), "
                         "train_size, d, n_rows -- the ROCm/MIGraphX contract (tabicl_mask.py).")
    args = ap.parse_args(argv)
    masked = args.contract == "mask"
    if masked and args.split_context:
        ap.error("--split-context is a positional-contract feature")

    cfg = configs.get(args.config, task=args.task)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"mask_tabicl_{args.task}" if masked else f"tabicl_{args.task}"
    graph_path = out / f"graph_{stem}.onnx"
    map_path = out / f"tensor_map_{stem}.json"

    print(f"[export_tabicl] building random-weight TabICL ({args.config}, {args.task}) ...",
          flush=True)
    t0 = time.time()
    model = build_model(args.task, cfg.model_kwargs, seed=args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export_tabicl] {n_params:,} params ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    if masked:
        wrapper = export.export_mask_graph(model, graph_path, dim_rows=cfg.dim_rows,
                                           dim_features=cfg.dim_features, example=cfg.example)
    else:
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
    if masked and not args.skip_parity:
        from export_tabicl.mask_parity import check_onnx
        t0 = time.time()
        parity = check_onnx(graph_path, model)
        print(f"[export_tabicl] masked ORT-vs-positional parity ({time.time()-t0:.1f}s): worst "
              f"{parity['worst']:.2e} -> {'OK' if parity['ok'] else 'FAIL'}", flush=True)
        if not parity["ok"]:
            print("[export_tabicl] PARITY FAILED", file=sys.stderr)
            return 1
    elif not args.skip_parity:
        t0 = time.time()
        parity = export.check_parity(graph_path, wrapper, cfg.parity_shapes)
        print(f"[export_tabicl] parity ({time.time()-t0:.1f}s): worst {parity['worst']:.2e} "
              f"(budget {parity['tol']:.0e}) -> {'OK' if parity['ok'] else 'FAIL'}", flush=True)
        if not parity["ok"]:
            print("[export_tabicl] PARITY FAILED", file=sys.stderr)
            return 1

    if args.split_context:
        prep_path = out / f"graph_prepare_tabicl_{args.task}.onnx"
        qry_path = out / f"graph_query_tabicl_{args.task}.onnx"
        t0 = time.time()
        split.export_split(model, prep_path, qry_path, dim_train=cfg.dim_train,
                           dim_features=cfg.dim_features, example=cfg.example)
        print(f"[export_tabicl] split export done ({time.time()-t0:.1f}s): "
              f"{split.n_context_tensors(model)} context tensor(s) + support_reps", flush=True)
        if not args.skip_parity:
            t0 = time.time()
            worst = split.check_split_parity(
                prep_path, qry_path, model,
                [(s_, max(2, t_ - s_), h_) for t_, h_, s_ in cfg.parity_shapes])
            ok = worst < 1e-3
            print(f"[export_tabicl] split parity ({time.time()-t0:.1f}s): worst {worst:.2e} "
                  f"(budget 1e-03) -> {'OK' if ok else 'FAIL'}", flush=True)
            if not ok:
                print("[export_tabicl] SPLIT PARITY FAILED", file=sys.stderr)
                return 1
        # Same shipping treatment as the single graph: externalize the checkpoint initializers,
        # write a tensor map per half, then drop the weights. Without this the pair would ship
        # random weights, which is exactly what assert_weight_free exists to prevent.
        for path, half in ((prep_path, "prepare"), (qry_path, "query")):
            tmap = export.postprocess(path, dict(model.state_dict()))
            # Named from its own graph by the rule the combined pair already
            # follows -- graph_ -> tensor_map_, .onnx -> .json -- so the runtime
            # finds a half's map from the half's filename and nothing else.
            export.write_tensor_map(out / f"tensor_map_{half}_tabicl_{args.task}.json", tmap,
                                    task=args.task,
                                    safetensors_rel=f"{args.task}/model.safetensors")
            export.delete_weight_data(path)
            export.assert_weight_free(path, tmap)
            print(f"[export_tabicl] split {half}: {path.name} "
                  f"({path.stat().st_size / 1e6:.2f} MB, weight-free, "
                  f"{len(tmap['initializers'])} initializers mapped)", flush=True)

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
    if masked:
        report["input_signature"] = {"x": "[1,T,H] f32 (padded)", "y": "[1,T] f32 FULL length (-100 off-context)",
                                     "train_size": "[1] i64", "d": "[1] i64 (real feature count)",
                                     "n_rows": "[1] i64 (declared, unused)"}
        report["train_size"] = "input (a value: the context is a mask, not a slice)"
        report["d"] = "input (padded feature groups masked in the row interactor)"
    (out / f"export_report_{stem}.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"[export_tabicl] graph: {graph_path} "
          f"({graph_path.stat().st_size/1e6:.2f} MB, weight-free)")
    print(f"[export_tabicl] map:   {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
