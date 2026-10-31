"""Build the random-init CI fixture in test/fixtures/tabpfn35/ (both tasks).

    uv run python make_fixture.py

Like the released checkpoints, ONE model and ONE `model.safetensors` serve both
tasks: the fixture model carries the 4-class decoder and the 16-bucket regression
head side by side, and the two graphs each map the subset they use. Everything is
seeded random-init (woken, so no projection is all-zero) -- NO Prior Labs weight
bytes. The regression borders are the architecture's own code-generated spline
borders for 16 buckets, not checkpoint data.

Per task: weight-free graph + tensor map + golden_<task>.json (fixed input ->
PyTorch logits on every row), self-checked by injecting the weights into ORT.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import torch
from safetensors.torch import save_file

from export_tabpfn import export as xp
from export_tabpfn35 import configs
from export_tabpfn35.cli import export_graph, injected_session, write_map
from export_tabpfn35.model import build_random_model
from export_tabpfn35.wrappers import Wrapper35, upstream_wrapped

SEED = 0
OUT = pathlib.Path(__file__).resolve().parents[2] / "test" / "fixtures" / "tabpfn35"
CFG = configs.get("fixture35")
T, H, N = CFG.example
ST = "model.safetensors"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = build_random_model(CFG.model_kwargs, seed=SEED)
    sd = dict(model.state_dict())
    written, maps = ["manifest.json"], {}
    for task in ("classification", "regression"):
        graph = OUT / f"graph_tabpfn35_{task}.onnx"
        tmap_path = OUT / f"tensor_map_tabpfn35_{task}.json"
        export_graph(model, graph, task=task, example=CFG.example, masked=False)
        maps[task] = xp.postprocess(graph, sd)
        write_map(tmap_path, maps[task], task=task, safetensors_rel=ST)
        xp.delete_weight_data(graph)
        xp.assert_weight_free(graph, maps[task])
        written += [graph.name, tmap_path.name]

    keys = sorted(set(maps["classification"]["initializers"].values())
                  | set(maps["regression"]["initializers"].values()))
    weights = {k: sd[k].detach().contiguous() for k in keys}
    save_file(weights, str(OUT / ST))
    written.append(ST)

    for task in ("classification", "regression"):
        g = torch.Generator().manual_seed(123)
        x = torch.randn(1, T, H, generator=g)
        if task == "classification":
            y = torch.arange(N, dtype=torch.float32).remainder(3)[None]
        else:
            y = torch.randn(1, N, generator=g) * 2.5 + 5.0
        with torch.no_grad():
            logits = Wrapper35(model, task=task)(x, y)
            up = upstream_wrapped(model, x, y, task)
        d_up = float((logits - up).abs().max())
        assert d_up < 1e-4, d_up

        sess = injected_session(OUT / f"graph_tabpfn35_{task}.onnx",
                                maps[task]["initializers"], weights)
        (r,) = sess.run(["logits"], {"x": x.numpy(), "y": y.numpy()})
        d = float(np.abs(r - logits.numpy()).max())
        assert d < 1e-3, d

        golden = OUT / ("golden.json" if task == "classification" else "golden_regression.json")
        golden.write_text(json.dumps({
            "_doc": {"purpose": f"{ST} -> initializer injection -> ORT on "
                                f"graph_tabpfn35_{task}.onnx reproduces these fp32 logits on EVERY row "
                                "(rows < N are the in-context fitted values).",
                     "torch_vs_upstream_max_abs": d_up, "ort_vs_torch_max_abs": d, "rtol": 1e-3},
            "task": task, "inputs": {"x": x.numpy().tolist(), "y": y.numpy().tolist(), "train_size": N},
            "logits": logits.numpy().tolist(),
        }, indent=2) + "\n")
        written.append(golden.name)
        print(f"  [{task}] torch-vs-upstream {d_up:.2e}, ORT-vs-torch {d:.2e}, "
              f"{len(maps[task]['initializers'])} mapped")

    st_bytes = (OUT / ST).stat().st_size
    manifest = {
        "schema_version": 2, "id": "tabpfn-v3-5-fixture",
        "display_name": "TabPFN 3.5 (random-init CI fixture)", "family": "icl-transformer",
        "license": {"id": "apache-2.0", "commercial": True, "redistributable": True,
                    "attribution": "Random-init fixture; no Prior Labs weights.", "gate_setting": None},
        "preprocessing_profile": "tabpfn_v3_5_raw",
        "weights": {t: {"files": [{"path": ST, "bytes": st_bytes}]}
                    for t in ("classification", "regression")},
        "graph": {"classification": "graph_tabpfn35_classification.onnx",
                  "regression": "graph_tabpfn35_regression.onnx",
                  "tensor_map": {t: f"tensor_map_tabpfn35_{t}.json"
                                 for t in ("classification", "regression")}},
        "capabilities": ["classify", "regress"],
        "tensor_contract": {
            "inputs": {"features": {"name": "x", "dtype": "f32", "shape": ["1", "T", "H"]},
                       "labels": {"name": "y", "dtype": "f32", "shape": ["1", "N"]}},
            "outputs": {"logits": {"name": "logits", "dtype": "f32", "shape": ["1", "T", "C"]}}},
        "size_regime": {"max_rows": 4096, "max_features": 64, "max_classes": 4},
        "compute": {"cpu": "f32"},
        "_note": ("Random-init TabPFN-3.5 (embed_dim=16, 2 ICL layers, ECDF buckets=6 so the "
                  "interpolating ruler runs). ONE safetensors serves both tasks. train_size is "
                  "len(y). See tools/export_tabpfn35."),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sums = [f"{hashlib.sha256((OUT / f).read_bytes()).hexdigest()}  {f}" for f in sorted(set(written))]
    (OUT / "FIXTURE_SHA256").write_text("\n".join(sums) + "\n")
    print("fixture written to", OUT, ":", ", ".join(sorted(set(written))))


if __name__ == "__main__":
    main()
