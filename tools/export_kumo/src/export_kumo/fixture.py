"""CI fixture model for Kumo-Tabular: random weights, tiny dims, deterministic bytes.

Same shape as the Causilo fixture (test/fixtures/causilo): a tiny random-init Kumo-Tabular (Apache-2.0
architecture, OUR weights, zero NVIDIA checkpoint bytes -- CLAUDE.md "license wall"), weight-free
graphs through the same exporter as the real sizes, and one v2 manifest carrying both tasks.

The golden inputs are exact in binary floating point and expressible in SQL, so the SQL test builds
the same table and asserts a prediction for EVERY row, context and query. Unlike Causilo's, Kumo's
graph takes ENGINE-STANDARDISED features (profile kumo_tabular_v1_minimal), so the golden values are
computed on `engine_ref.engine_inputs(...)`, the float64 replica of the engine's preprocessing; the
SQL test therefore also checks that replica against the engine.

Conditioning, measured rather than guessed (the two failure modes the Causilo fixture documents).
Upstream zero-initialises every residual output projection, so a default random model is the
identity on its residual stream; `build_model` wakes them (WAKE_STD). Even then a 12-row context
gives LogScale a temperature of a_h * log(12) with a_h in [0, 1): attention is nearly uniform and one
context label moves the query outputs by ~5e-2. Turning up the length scale (ATTENTION_GAIN) and the
last head layer (HEAD_GAIN) gives label sensitivity well above MIN_LABEL_SENSITIVITY while a 1e-6
relative input change still moves the output by far less than MAX_NOISE_RESPONSE. Turning up the label
embeddings instead does nothing: every stage RMS-normalises its input. Seed 13 is the first that
predicts all three classes with a top-1/top-2 margin above MIN_MARGIN on every row.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import torch
from safetensors.torch import save_file

from export_kumo import configs as k_configs
from export_kumo import engine_ref
from export_kumo import export as k_export
from export_kumo.kumo_patches import apply, build_model

SEED_WEIGHTS = 13
WAKE_STD = 0.5                             # the residual projections upstream zero-initialises
ATTENTION_GAIN = 20.0                      # LogScale / GatedLogScale head_scale: sharper attention
HEAD_GAIN = 10.0                           # last ICL head layer: logits well clear of rounding
MIN_LABEL_SENSITIVITY = 0.1                # query-output change when one context label changes
MAX_NOISE_RESPONSE = 1e-3                  # output change per 1e-6 relative input perturbation
GOLDEN_SHAPE = dict(t=20, h=5, s=12)
PARITY_RTOL = 1e-4
MIN_MARGIN = 0.05

TASKS = ("classification", "regression")
FILES = [
    "graph_kumo_fixture_classification.onnx", "model_classification.safetensors",
    "tensor_map_kumo_fixture_classification.json", "golden_classification.json",
    "graph_kumo_fixture_regression.onnx", "model_regression.safetensors",
    "tensor_map_kumo_fixture_regression.json", "golden_regression.json",
    "manifest.json",
]


def sql_inputs(task: str):
    """(x [T,H], y [S]) as nested lists, from formulas the SQL test repeats verbatim.

    x[i][j] = ((i*7 + j*3 + (i*j) % 5) % 11) * 0.25 - 1.0     exact in float32
    classification y[i] = (i*5 + i//3) % 3                     labels 'c0','c1','c2'
    regression     y[i] = ((i*37) % 17) * 0.25 - 2.0           exact in float32
    """
    t, h, s = (GOLDEN_SHAPE[k] for k in ("t", "h", "s"))
    x = [[((i * 7 + j * 3 + (i * j) % 5) % 11) * 0.25 - 1.0 for j in range(h)] for i in range(t)]
    if task == "classification":
        y = [float((i * 5 + i // 3) % 3) for i in range(s)]
    else:
        y = [((i * 37) % 17) * 0.25 - 2.0 for i in range(s)]
    return x, y


def seeded_model(task: str):
    cfg = k_configs.get("fixture")
    model = build_model(task, cfg.model, seed=SEED_WEIGHTS, num_classes=cfg.classes,
                        num_quantiles=cfg.quantiles, wake_std=WAKE_STD)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("head_scale"):
                param.mul_(ATTENTION_GAIN)
            elif name.startswith("icl_block.head.2."):
                param.mul_(HEAD_GAIN)
    return model


def _feed(task: str):
    xs, ys = sql_inputs(task)
    feed, inv = engine_ref.engine_inputs(np.array(xs), np.array(ys),
                                         regression=task == "regression")
    return {k: torch.from_numpy(v) for k, v in feed.items()}, inv


def _call(wrapper, feed):
    return wrapper(feed["x"], feed["y"], feed["train_size"], feed["cat_mask"], feed["d"])


def _assert_well_conditioned(wrapper, feed, task: str) -> None:
    s = int(feed["train_size"][0])
    with torch.no_grad():
        base = _call(wrapper, feed)
        noise = 0.0
        for k in range(3):
            g = torch.Generator().manual_seed(k)
            f2 = dict(feed)
            f2["x"] = feed["x"] * (1 + 1e-6 * torch.randn(feed["x"].shape, generator=g))
            noise = max(noise, (_call(wrapper, f2) - base).abs().max().item())
        f3 = dict(feed)
        f3["y"] = feed["y"].clone()
        f3["y"][0, 0] = (f3["y"][0, 0] + 1) % 3 if task == "classification" else f3["y"][0, 0] + 1.5
        label = (_call(wrapper, f3)[0, s:] - base[0, s:]).abs().max().item()
    assert noise < MAX_NOISE_RESPONSE, f"ill-conditioned fixture: noise response {noise:.2e}"
    assert label > MIN_LABEL_SENSITIVITY, f"label-blind fixture: label change moves output {label:.2e}"


def safetensors_digest(model, path: pathlib.Path) -> str:
    sd = {k: v.detach().contiguous().clone() for k, v in sorted(model.state_dict().items())}
    save_file(sd, str(path),
              metadata={"origin": "anofox-tabfm CI fixture, random init, Apache-2.0 Kumo-Tabular "
                                  "architecture (our weights, not NVIDIA's)"})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _build_task(out: pathlib.Path, task: str) -> str:
    model, model2 = seeded_model(task), seeded_model(task)
    sd, sd2 = model.state_dict(), model2.state_dict()
    assert sorted(sd) == sorted(sd2)
    for k in sd:
        assert torch.equal(sd[k], sd2[k]), f"nondeterministic weight {k}"

    st_path = out / f"model_{task}.safetensors"
    digest = safetensors_digest(model, st_path)
    tmp = out / f"model_{task}.safetensors.recheck"
    digest2 = safetensors_digest(model2, tmp)
    tmp.unlink()
    assert digest == digest2, f"safetensors bytes not deterministic: {digest} != {digest2}"

    cfg = k_configs.get("fixture")
    regression = task == "regression"
    classes = 0 if regression else cfg.classes
    graph_path = out / f"graph_kumo_fixture_{task}.onnx"
    wrapper = k_export.export_graph(model, graph_path, example=cfg.example, classes=classes,
                                    regression=regression)
    tensor_map = k_export.postprocess(graph_path, dict(model.state_dict()))
    k_export.write_tensor_map(out / f"tensor_map_kumo_fixture_{task}.json", tensor_map,
                              task=task, safetensors_rel=f"model_{task}.safetensors")
    parity = k_export.check_parity(graph_path, wrapper, cfg.parity_shapes, classes)
    assert parity["ok"], parity
    k_export.delete_weight_data(graph_path)
    k_export.assert_weight_free(graph_path, tensor_map)

    feed, (mean, std) = _feed(task)
    _assert_well_conditioned(wrapper, feed, task)
    with torch.no_grad():
        logits = _call(wrapper, feed)                                # [1, T, C]
    per_row = {}
    if task == "classification":
        top2 = logits[0, :, :cfg.classes].topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).min().item()
        assert margin > MIN_MARGIN, f"a golden row is a near-tie (min margin {margin:.3g})"
        labels = logits[0, :, :cfg.classes].argmax(-1).tolist()
        assert len(set(labels)) == cfg.classes, f"golden labels do not cover every class: {labels}"
        per_row["labels"] = [f"c{int(c)}" for c in labels]
    else:
        values = logits[0, :, 0].double() * std + mean               # the engine's inverse z-score
        assert values.std().item() > 0.05, "golden regression output is nearly constant"
        per_row["values"] = [round(v, 6) for v in values.tolist()]

    (out / f"golden_{task}.json").write_text(json.dumps({
        "_doc": {
            "purpose": f"Parity: safetensors -> initializer injection -> ORT on "
                       f"graph_kumo_fixture_{task}.onnx must reproduce these outputs for EVERY row "
                       f"(rows < train_size are in-context fitted values).",
            "rtol": PARITY_RTOL,
            "inputs": "engine_ref.engine_inputs(sql_inputs(task)): the engine's float64 "
                      "standardisation of x[i][j] = ((i*7 + j*3 + (i*j) % 5) % 11) * 0.25 - 1.0; "
                      + ("y[i] = (i*5 + i//3) % 3" if task == "classification"
                         else "y[i] = ((i*37) % 17) * 0.25 - 2.0, z-scored; per_row values are "
                              "de-standardised (yhat = out * std + mean) as the engine reports them"),
        },
        "inputs": {k: v.tolist() for k, v in feed.items()},
        "outputs": logits.tolist(),
        "output_shape": list(logits.shape),
        "per_row": per_row,
        "safetensors_sha256": digest,
    }, indent=2) + "\n")
    return digest


def build(out: pathlib.Path) -> dict:
    """Build the dual-task committed CI fixture (both tasks + combined manifest)."""
    apply()
    out.mkdir(parents=True, exist_ok=True)
    for t in TASKS:
        _build_task(out, t)

    def file_entry(name):
        p = out / name
        return {"path": name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}

    manifest = {
        "schema_version": 2, "id": "kumo-fixture",
        "display_name": "Kumo-Tabular CI fixture (random init, schema v2)",
        "family": "icl-transformer",
        "license": {"id": "apache-2.0", "commercial": True, "redistributable": True,
                    "gate_setting": None},
        "preprocessing_profile": "kumo_tabular_v1_minimal",
        "weights": {
            t: {"repo": "local:test/fixtures/kumo", "revision": "fixture-v1",
                "files": [file_entry(f"model_{t}.safetensors")]}
            for t in TASKS
        },
        "graph": {"classification": "graph_kumo_fixture_classification.onnx",
                  "regression": "graph_kumo_fixture_regression.onnx",
                  "tensor_map": {"classification": "tensor_map_kumo_fixture_classification.json",
                                 "regression": "tensor_map_kumo_fixture_regression.json"}},
        "capabilities": ["classify", "regress"],
        "tensor_contract": {
            "inputs": {
                "features": {"name": "x", "dtype": "f32", "shape": ["1", "T", "H"]},
                "labels": {"name": "y", "dtype": "f32", "shape": ["1", "T"]},
                "train_size": {"name": "train_size", "dtype": "i64"},
                "n_features": {"name": "d", "dtype": "i64"},
            },
            "outputs": {"logits": {"name": "logits", "dtype": "f32", "shape": ["1", "T", "C"]}},
        },
        "size_regime": {"max_rows": 4096, "max_features": 64, "max_classes": 3},
        "compute": {"cpu": "f32"},
        "_note": "Random-init Kumo-Tabular fixture (Apache-2.0 architecture, our weights). T and H "
                 "are DYNAMIC; the tabfm-v1 five-input contract (x, y[1,T], train_size, cat_mask, "
                 "d); the engine standardises features and the regression target "
                 "(kumo_tabular_v1_minimal). Classification outputs are class logits (C = 3); "
                 "regression outputs the mean of 16 quantiles (C = 1).",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    hashes = {name: sha256_file(out / name) for name in FILES}
    total = sum((out / name).stat().st_size for name in FILES)
    assert total < 5 * 1024 * 1024, f"fixture total {total} B >= 5 MB budget"
    return hashes
