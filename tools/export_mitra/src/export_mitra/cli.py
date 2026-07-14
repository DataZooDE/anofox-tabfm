"""CLI: uv run export_mitra --task classification --config real --out DIR.

Writes into --out:
  graph_mitra_<task>.onnx        weight-free graph (checkpoint initializers
                                 EXTERNAL stubs; the .onnx.data is deleted)
  tensor_map_mitra_<task>.json   ONNX initializer name -> safetensors key
  export_report_mitra_<task>.json  provenance + parity numbers

Fixture extras (opt-in):
  --emit-weights PATH   save the random-init state_dict as a tiny safetensors
  --emit-golden PATH    save a golden parity slice (fixed feed + ORT logits)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time

import numpy as np

from export_mitra import configs, export


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="export_mitra")
    ap.add_argument("--task", required=True,
                    choices=["classification", "regression"])
    ap.add_argument("--config", required=True, choices=["real", "fixture"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--emit-weights", default=None)
    ap.add_argument("--emit-golden", default=None)
    args = ap.parse_args(argv)

    cfg = configs.get(args.config, args.task)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / f"graph_mitra_{args.task}.onnx"
    map_path = out / f"tensor_map_mitra_{args.task}.json"

    print(f"[export_mitra] building random-weight Tab2D ({args.config}, "
          f"{args.task}) ...", flush=True)
    t0 = time.time()
    model = export.build_model(args.task, cfg.model_kwargs, seed=args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export_mitra] {n_params:,} params ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    wrapper = export.export_graph(
        model, graph_path, dim_rows=cfg.dim_rows,
        dim_features=cfg.dim_features, example=cfg.example)
    print(f"[export_mitra] dynamo export done ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    state_dict = {k: v for k, v in model.state_dict().items()}
    tensor_map = export.postprocess(graph_path, state_dict)
    print(f"[export_mitra] postprocess done ({time.time()-t0:.1f}s): "
          f"{len(tensor_map['initializers'])} initializers mapped, "
          f"{len(tensor_map['unmatched_small'])} small inline constants",
          flush=True)
    export.write_tensor_map(
        map_path, tensor_map, task=args.task,
        safetensors_rel=f"{args.task}/model.safetensors")

    parity = None
    if not args.skip_parity:
        t0 = time.time()
        parity = export.check_parity(graph_path, model, wrapper, cfg.parity_shapes)
        print(f"[export_mitra] parity ({time.time()-t0:.1f}s): worst "
              f"{parity['worst']:.2e} (budget {parity['tol']:.0e}) -> "
              f"{'OK' if parity['ok'] else 'FAIL'}", flush=True)
        if not parity["ok"]:
            print("[export_mitra] PARITY FAILED", file=sys.stderr)
            return 1

    # optional: random-init safetensors for the CI fixture (before we drop .data)
    if args.emit_weights:
        from safetensors.torch import save_file
        wp = pathlib.Path(args.emit_weights)
        wp.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in state_dict.items()}, str(wp))
        print(f"[export_mitra] wrote fixture weights -> {wp} "
              f"({wp.stat().st_size} bytes)", flush=True)

    # optional: golden parity slice (fixed feed + ORT logits on test rows)
    if args.emit_golden:
        import onnxruntime as ort
        n_classes = model.dim_output if args.task == "classification" else 1
        feed = export.make_feed(*cfg.example[:1], cfg.example[1], cfg.example[2],
                                cfg.example[3], n_classes=max(n_classes, 2), seed=123)
        sess = ort.InferenceSession(str(graph_path),
                                    providers=["CPUExecutionProvider"])
        (logits,) = sess.run(["logits"], feed)
        train = int(feed["train_size"][0])
        gp = pathlib.Path(args.emit_golden)
        gp.parent.mkdir(parents=True, exist_ok=True)
        golden = {
            "_doc": {
                "purpose": "C++ parity: random-init model.safetensors -> "
                           "initializer injection (via tensor_map) -> ORT run "
                           "on the weight-free fixture graph must reproduce "
                           "these fp32 logits.",
                "parity_slice": "logits[:, train_size:, :] (test rows); the "
                                "engine reads predictions only there.",
                "cat_mask": "OMITTED — Mitra has no categorical embedding; the "
                            "contract is (x, y, train_size, d) -> logits.",
                "rtol": 1e-4,
            },
            "task": args.task,
            "inputs": {
                "x": feed["x"].astype(float).round(6).tolist(),
                "y": feed["y"].astype(float).tolist(),
                "train_size": train,
                "d": int(feed["d"][0]),
            },
            "logits": logits[0].astype(float).round(6).tolist(),
            "logits_test_rows": logits[0, train:].astype(float).round(6).tolist(),
        }
        gp.write_text(json.dumps(golden, indent=2) + "\n")
        print(f"[export_mitra] wrote golden -> {gp}", flush=True)

    report = {
        "model": "mitra-tab2d",
        "task": args.task,
        "config": args.config,
        "opset": export.OPSET,
        "n_params": n_params,
        "model_kwargs": cfg.model_kwargs,
        "input_signature": {
            "x": "[1,T,H] f32", "y": "[1,T] f32",
            "train_size": "[1] i64", "d": "[1] i64"},
        "output": {"logits": "[1,T,C] f32"},
        "cat_mask": "omitted (Mitra has no categorical embedding)",
        "graph_bytes": graph_path.stat().st_size,
        "initializers_mapped": len(tensor_map["initializers"]),
        "unmatched_small_inline": tensor_map["unmatched_small"],
        "parity": parity,
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__},
        "cmd": "export_mitra " + " ".join(sys.argv[1:]),
    }
    try:
        import torch, onnx, onnxruntime
        report["versions"].update(torch=torch.__version__, onnx=onnx.__version__,
                                  onnxruntime=onnxruntime.__version__)
    except Exception:
        pass

    export.delete_weight_data(graph_path)
    export.assert_weight_free(graph_path, tensor_map)
    report["weight_free"] = True
    (out / f"export_report_mitra_{args.task}.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"[export_mitra] DONE -> {graph_path} (weight-free), {map_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
