"""CLI: uv run export_causilo --task classification --config real --out DIR.

Writes into --out:
  graph_causilo_<task>.onnx          weight-free graph (checkpoint initializers are EXTERNAL
                                     stubs; the .onnx.data is deleted)
  graph_mask_causilo_<task>.onnx     with --contract mask: the MASKED graph (x, y, train_size, n_rows, d)
                                     that MIGraphX can compile per shape bucket; docs/ROCM_CAUSILO_PLAN.md
  tensor_map_causilo_<task>.json     ONNX initializer name -> safetensors key
  export_report_causilo_<task>.json  provenance + parity numbers
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from export_causilo import configs, export
from export_causilo.causilo_patches import build_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="export_causilo")
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--config", required=True, choices=["fixture", "real"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--contract", choices=["positional", "mask"], default="positional",
                    help="positional: the split is the LENGTH of y (the shipped CPU/CUDA form). mask: y is "
                         "full length and the split, real row count and real feature count arrive as values, "
                         "so a fixed-shape MIGraphX compile can bucket it.")
    args = ap.parse_args(argv)
    masked = args.contract == "mask"

    cfg = configs.get(args.config)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / (f"graph_mask_causilo_{args.task}.onnx" if masked else f"graph_causilo_{args.task}.onnx")
    map_path = out / f"tensor_map_causilo_{args.task}.json"
    classes = cfg.classes if args.task == "classification" else 0

    print(f"[export_causilo] building random-weight Causilo ({args.config}, {args.task}) ...", flush=True)
    t0 = time.time()
    model = build_model(args.task, cfg, seed=args.seed)
    if masked:
        # Weights are random either way (the graph is weight-free), but a DEFAULT-initialised Causilo ignores
        # its input (measured at real dims: classification logits vary by 0.019 across rows), so a parity check
        # against it passes vacuously. sensitive() makes it respond to its labels and features without making
        # it ill-conditioned (1e-6 relative input noise moves the output <= 3e-4 at real dims).
        from export_causilo.fixture import sensitive
        model = sensitive(model, weight_gain=cfg.weight_gain)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export_causilo] {n_params:,} params ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    if masked:
        export.export_mask_graph(model, graph_path, example=cfg.mask_example, classes=classes)
        wrapper = None
    else:
        wrapper = export.export_graph(model, graph_path, example=cfg.example, classes=classes)
    print(f"[export_causilo] dynamo export done ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    tensor_map = export.postprocess(graph_path, dict(model.state_dict()))
    print(f"[export_causilo] postprocess done ({time.time()-t0:.1f}s): "
          f"{len(tensor_map['initializers'])} initializers mapped, "
          f"{len(tensor_map['unmatched_small'])} small inline constants", flush=True)
    export.write_tensor_map(map_path, tensor_map, task=args.task,
                            safetensors_rel=f"{args.task}/model.safetensors")

    parity = None
    if masked and not args.skip_parity:
        from export_causilo.causilo_patches import ExportWrapper
        t0 = time.time()
        parity = export.check_mask_parity(graph_path, ExportWrapper(model).eval(), cfg.mask_parity_cases, classes)
        print(f"[export_causilo] masked parity vs the UNPADDED wrapper ({time.time()-t0:.1f}s): worst "
              f"{parity['worst']:.2e} (budget {parity['tol']:.0e}, every row) -> {'OK' if parity['ok'] else 'FAIL'}",
              flush=True)
        if not parity["ok"]:
            print("[export_causilo] MASKED PARITY FAILED", file=sys.stderr)
            return 1
    elif not args.skip_parity:
        t0 = time.time()
        parity = export.check_parity(graph_path, wrapper, cfg.parity_shapes, classes)
        print(f"[export_causilo] parity ({time.time()-t0:.1f}s): worst {parity['worst']:.2e} "
              f"(budget {parity['tol']:.0e}, every row) -> {'OK' if parity['ok'] else 'FAIL'}", flush=True)
        if not parity["ok"]:
            print("[export_causilo] PARITY FAILED", file=sys.stderr)
            return 1

    export.delete_weight_data(graph_path)
    export.assert_weight_free(graph_path, tensor_map)

    import onnx as _onnx
    import onnxruntime as _ort
    import onnxscript as _onnxscript
    import torch as _torch
    report = {
        "command": ["export_causilo"] + list(argv or sys.argv[1:]),
        "task": args.task, "config": cfg.name, "model": cfg.model,
        "outputs": cfg.classes if args.task == "classification" else cfg.channels,
        "n_params": n_params, "seed": args.seed, "opset": export.OPSET,
        "contract": args.contract,
        "input_signature": (
            {"x": "[1,T,H] f32: engine-preprocessed (encoded, finite), NOT normalised; real data in [:n_rows,:d], zeros elsewhere",
             "y": "[1,T] f32: training targets in [:train_size], -100 elsewhere; class ids, or RAW regression targets",
             "train_size": "[1] int64", "n_rows": "[1] int64 (declared for plugin parity; unused)", "d": "[1] int64"}
            if masked else
            {"x": "[1,T,H] f32: engine-preprocessed (encoded, finite), NOT normalised",
             "y": "[1,S] f32: training targets only (S = train_size); class ids, or RAW regression targets"}),
        "output": (
            {"logits": "[1,T,C] class logits, C=10; rows < S are in-context fitted values "
                       "(context rows presented a second time as queries)"}
            if args.task == "classification" else
            {"logits": "[1,T,1] RAW-space point estimate per row: the mean over the 999 output "
                       "channels, inverse target StandardScaler applied in-graph; rows < S are "
                       "in-context fitted values. The engine feeds RAW train targets and reads "
                       "the output directly."}),
        "in_graph_preprocessing": "Causilo single-estimator ('none') normaliser: train-prefix "
                                  "z-score, two-stage 4-sigma bounds, arcsinh tails; NaN restored",
        "split": ("value: train_size / n_rows / d inputs; y is full length" if masked
                  else "positional: S = len(y); no train_size, cat_mask or d input"),
        "graph_bytes": graph_path.stat().st_size,
        "n_initializers_mapped": len(tensor_map["initializers"]),
        "unmatched_small_inline": tensor_map["unmatched_small"],
        "parity": parity,
        "versions": {"torch": _torch.__version__, "onnx": _onnx.__version__,
                     "onnxruntime": _ort.__version__, "onnxscript": _onnxscript.__version__},
    }
    report_name = f"export_report_mask_causilo_{args.task}.json" if masked else f"export_report_causilo_{args.task}.json"
    (out / report_name).write_text(json.dumps(report, indent=2) + "\n")

    print(f"[export_causilo] graph: {graph_path} "
          f"({graph_path.stat().st_size/1e6:.2f} MB, weight-free)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
