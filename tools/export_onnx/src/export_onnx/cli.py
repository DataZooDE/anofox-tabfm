"""CLI: uv run export_onnx --task classification --config tiny --out DIR.

Writes into --out:
  graph_<task>.onnx        weight-free graph (checkpoint initializers are
                           EXTERNAL stubs; the .onnx.data is deleted)
  tensor_map_<task>.json   ONNX initializer name -> safetensors key
  export_report_<task>.json  provenance: versions, config source + sha,
                           parity numbers, sizes, command line
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from export_onnx import configs, export


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(prog="export_onnx")
  ap.add_argument("--task", required=True,
                  choices=["classification", "regression"])
  ap.add_argument("--config", required=True,
                  choices=["tiny", "fixture", "real"])
  ap.add_argument("--out", required=True)
  ap.add_argument("--seed", type=int, default=0,
                  help="seed for the random weights (graph is weight-free; "
                       "only parity depends on it)")
  ap.add_argument("--skip-parity", action="store_true",
                  help="skip the ORT-vs-PyTorch parity check")
  ap.add_argument("--config-json", default=None,
                  help="local config.json for --config real (skips the "
                       "anonymous HF fetch)")
  args = ap.parse_args(argv)

  cfg = configs.get(args.config, args.task, args.config_json)
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  graph_path = out / f"graph_{args.task}.onnx"
  map_path = out / f"tensor_map_{args.task}.json"

  print(f"[export_onnx] building random-weight TabFM ({args.config}, "
        f"{args.task}) ...", flush=True)
  t0 = time.time()
  model = export.build_model(args.task, cfg.model_kwargs, seed=args.seed)
  n_params = sum(p.numel() for p in model.parameters())
  print(f"[export_onnx] {n_params:,} params ({time.time()-t0:.1f}s)",
        flush=True)

  t0 = time.time()
  export.export_graph(model, graph_path, dim_rows=cfg.dim_rows,
                      dim_features=cfg.dim_features, example=cfg.example)
  print(f"[export_onnx] dynamo export done ({time.time()-t0:.1f}s)",
        flush=True)

  t0 = time.time()
  tensor_map = export.postprocess(graph_path, dict(model.state_dict()))
  print(f"[export_onnx] postprocess done ({time.time()-t0:.1f}s): "
        f"{len(tensor_map['initializers'])} initializers mapped, "
        f"{len(tensor_map['unmatched_small'])} small inline constants",
        flush=True)
  export.write_tensor_map(
      map_path, tensor_map, task=args.task,
      safetensors_rel=f"{args.task}/model.safetensors")

  parity = None
  if not args.skip_parity:
    t0 = time.time()
    parity = export.check_parity(graph_path, model, cfg.parity_shapes)
    print(f"[export_onnx] parity ({time.time()-t0:.1f}s): worst "
          f"{parity['worst']:.2e} (budget {parity['tol']:.0e}) -> "
          f"{'OK' if parity['ok'] else 'FAIL'}", flush=True)
    if not parity["ok"]:
      print("[export_onnx] PARITY FAILED", file=sys.stderr)
      return 1

  export.delete_weight_data(graph_path)
  export.assert_weight_free(graph_path, tensor_map)

  import onnx as _onnx
  import onnxruntime as _ort
  import onnxscript as _onnxscript
  import torch as _torch
  report = {
      "command": ["export_onnx"] + list(argv or sys.argv[1:]),
      "task": args.task,
      "config": cfg.name,
      "config_source": cfg.config_source,
      "model_kwargs": cfg.model_kwargs,
      "n_params": n_params,
      "seed": args.seed,
      "opset": export.OPSET,
      "dim_rows": list(cfg.dim_rows),
      "dim_features": list(cfg.dim_features),
      "example_input_THdtrain": list(cfg.example),
      "graph_bytes": graph_path.stat().st_size,
      "n_initializers_mapped": len(tensor_map["initializers"]),
      "n_transforms": len(tensor_map["transforms"]),
      "unmatched_small_inline": tensor_map["unmatched_small"],
      "parity": parity,
      "versions": {
          "torch": _torch.__version__,
          "onnx": _onnx.__version__,
          "onnxruntime": _ort.__version__,
          "onnxscript": _onnxscript.__version__,
      },
  }
  report_path = out / f"export_report_{args.task}.json"
  report_path.write_text(json.dumps(report, indent=2) + "\n")

  print(f"[export_onnx] graph: {graph_path} "
        f"({graph_path.stat().st_size/1e6:.2f} MB, weight-free)")
  print(f"[export_onnx] map:   {map_path}")
  print(f"[export_onnx] report: {report_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
