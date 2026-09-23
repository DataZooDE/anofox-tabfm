"""TabDPT -> weight-free ONNX graph + tensor map.

  uv run export_tabdpt --task classification --config real    --out ../../resources
  uv run export_tabdpt --task classification --config fixture --out ./out

Unlike every other exporter here there is NO companion convert_weights.py step:
Layer 6 publish the checkpoint as safetensors already
(`Layer6/TabDPT :: tabdpt1_2.safetensors`, Apache-2.0, ungated) and its keys are
already the model's state_dict namespace, so the engine injects the downloaded
file straight against the committed tensor map.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

from export_tabdpt import configs, export
from export_tabdpt.tabdpt_patches import build_wrapper

SLUG = "tabdpt"


def build_model(cfg: configs.ExportConfig, seed: int = 0):
    """Random-init TabDPTModel at the config's dims. No checkpoint bytes."""
    from tabdpt.model import TabDPTModel

    torch.manual_seed(seed)
    return TabDPTModel(**cfg.model_kwargs).eval()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="export_tabdpt")
    ap.add_argument("--task", required=True,
                    choices=["classification", "regression"])
    ap.add_argument("--config", required=True, choices=["real", "fixture"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--contract", choices=["positional", "mask"], default="positional",
                    help="positional: y is the train prefix and the split is its LENGTH (the shipped "
                         "CPU/CUDA form). mask: y is full length and the split arrives as a train_size "
                         "VALUE, so every shape depends on the (T, H) bucket alone -- the only form "
                         "MIGraphX can compile. See docs/ROCM_TABDPT_SPIKE.md.")
    ap.add_argument("--weights",
                    help="published safetensors; when given, configs.real() is "
                         "checked against its embedded cfg metadata")
    args = ap.parse_args(argv)

    cfg = configs.get(args.config)
    if args.config == "real" and args.weights:
        configs.assert_matches_checkpoint(args.weights)
        print(f"[export_tabdpt] config matches {args.weights}", flush=True)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    masked = args.contract == "mask"
    stem = f"graph_mask_{SLUG}" if masked else f"graph_{SLUG}"
    graph_path = out / f"{stem}_{args.task}.onnx"
    map_path = out / f"tensor_map_{SLUG}_{args.task}.json"

    print(f"[export_tabdpt] building random-weight TabDPT "
          f"({args.config}, {args.task}) ...", flush=True)
    t0 = time.time()
    model = build_model(cfg, seed=args.seed)
    if masked:
        from export_tabdpt.tabdpt_mask_patches import apply_migraphx_friendly, build_mask_wrapper
        # chunk -> slices: MIGraphX's parser implements neither SplitToSequence
        # nor SequenceAt, and TabDPT's SwiGLU emits 32 of the first and 64 of
        # the second.
        apply_migraphx_friendly()
        wrapper = build_mask_wrapper(model, args.task)
    else:
        wrapper = build_wrapper(model, args.task)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export_tabdpt] {n_params:,} params, n_out={model.n_out}, "
          f"num_features={model.num_features} ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    if masked:
        export.export_mask_graph(wrapper, graph_path, cfg=cfg)
    else:
        export.export_graph(wrapper, graph_path, cfg=cfg)
    print(f"[export_tabdpt] dynamo export done ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    tensor_map = export.postprocess(graph_path, model.state_dict())
    export.write_tensor_map(map_path, tensor_map, task=args.task,
                            safetensors_rel=f"{args.task}/model.safetensors")
    print(f"[export_tabdpt] postprocess ({time.time() - t0:.1f}s): "
          f"{len(tensor_map['initializers'])} initializers mapped, "
          f"{len(tensor_map['unmatched_small'])} small inline constants", flush=True)

    if not args.skip_parity:
        t0 = time.time()
        if masked:
            # check_parity feeds (x, y) only, which is the POSITIONAL contract;
            # the masked graph declares train_size and d as well, so the
            # positional check fails with "Required inputs are missing" rather
            # than telling you anything about the graph. Route to the masked
            # gate, which also runs its negative controls first.
            from export_tabdpt.mask_parity import main as mask_gate
            rc = mask_gate(["--config", args.config])
            print(f"[export_tabdpt] masked parity gate ({time.time() - t0:.1f}s): "
                  f"{'OK' if rc == 0 else 'FAIL'}", flush=True)
            if rc != 0:
                return rc
            rep = None
        else:
            rep = export.check_parity(graph_path, wrapper, cfg.parity_shapes,
                                      tol=cfg.parity_tol)
        if rep is not None:
            print(f"[export_tabdpt] parity ({time.time() - t0:.1f}s): "
                  f"worst {rep['worst']:.2e} (budget {cfg.parity_tol:g}) -> "
                  f"{'OK' if rep['ok'] else 'FAIL'}", flush=True)
            if not rep["ok"]:
                return 1

    export.delete_weight_data(graph_path)
    export.assert_weight_free(graph_path, tensor_map)
    size_mb = graph_path.stat().st_size / 1e6
    print(f"[export_tabdpt] graph: {graph_path} ({size_mb:.2f} MB, weight-free)",
          flush=True)

    report = out / f"export_report_{SLUG}_{args.task}.json"
    report.write_text(json.dumps({
        "model": "tabdpt", "task": args.task, "config": cfg.name,
        "torch": torch.__version__,
        "n_params": n_params, "n_mapped": len(tensor_map["initializers"]),
        "graph_bytes": graph_path.stat().st_size,
        "command": " ".join(sys.argv),
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
