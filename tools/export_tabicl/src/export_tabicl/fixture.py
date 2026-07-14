"""CI fixture model for TabICL — random weights, tiny dims, deterministic bytes.

Same shape as the S06 TabFM fixture (tools/make_fixture) and the Mitra fixture
(test/fixtures/mitra): a tiny random-init TabICL (BSD-3-Clause architecture, our
weights — zero soda-inria checkpoint bytes) with SEEDED weights over SORTED
state_dict keys, weight-free graphs through the SAME exporter, per-task
golden_*.json (PyTorch fp32 logits for the C++ parity test) and a single v2
manifest carrying BOTH tasks (capabilities [classify, regress]).

Regression is a real second capability here: the exported regression graph emits
``logits[1, T, 1]`` — a single real-valued POINT ESTIMATE per row (mean over the
quantile head, then an in-graph inverse StandardScaler), NOT 999 quantile logits.
See ``tabicl_patches.ExportWrapper`` for the target-space contract.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import torch
from safetensors.torch import save_file

from export_tabicl import configs as x_configs
from export_tabicl import export as x_export
from export_tabicl.tabicl_patches import ExportWrapper, apply, build_model

SEED_WEIGHTS = 1337
SEED_GOLDEN_INPUTS = 7
WEIGHT_SCALE = 0.05
GOLDEN_SHAPE = dict(t=20, h=5, s=12)  # T rows, H features, S train_size
PARITY_RTOL = 1e-4

TASKS = ("classification", "regression")
# Shared tensor map: injection is name-based (onnx init "m.<key>" <- safetensors
# "<key>", with an "m."-prefix fallback in the engine), so one map serves both
# graphs. We ship the classification map as the manifest's shared map.
SHARED_TENSOR_MAP = "tensor_map_tabicl_classification.json"
FILES = [
    "graph_tabicl_classification.onnx", "model_classification.safetensors",
    "tensor_map_tabicl_classification.json", "golden_classification.json",
    "graph_tabicl_regression.onnx", "model_regression.safetensors",
    "tensor_map_tabicl_regression.json", "golden_regression.json",
    "manifest.json",
]
# Stale single-task fixture artifacts from before the dual-task restructure.
_LEGACY = ["graph_tabicl_fixture.onnx", "model.safetensors",
           "tensor_map_tabicl_fixture.json", "golden.json"]


def seeded_model(task: str = "classification"):
    cfg = x_configs.fixture()
    model = build_model(task, cfg.model_kwargs, seed=0)
    gen = torch.Generator().manual_seed(SEED_WEIGHTS)
    with torch.no_grad():
        sd = model.state_dict()
        for key in sorted(sd):  # sorted: init-order independent
            p = sd[key]
            if p.dtype.is_floating_point:
                p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * WEIGHT_SCALE)
    model.train()
    return model


def safetensors_bytes_digest(model, st_path: pathlib.Path) -> str:
    save_file(
        model.state_dict(), str(st_path),
        metadata={"origin": "anofox-tabfm CI fixture, random init, BSD-3-Clause "
                            "TabICL architecture (our weights, not soda-inria's)"})
    return hashlib.sha256(st_path.read_bytes()).hexdigest()


def golden_inputs(task: str = "classification"):
    gen = torch.Generator().manual_seed(SEED_GOLDEN_INPUTS)
    t, h, s = (GOLDEN_SHAPE[k] for k in ("t", "h", "s"))
    x = torch.randn(1, t, h, generator=gen)
    if task == "classification":
        y = torch.randint(0, 3, (1, s), generator=gen).float()
    else:
        # RAW train targets (the engine feeds raw; the graph z-scores internally).
        y = torch.randn(1, s, generator=gen) * 2.5 + 1.0
    return x, y


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _build_task(out: pathlib.Path, task: str) -> str:
    """Build one task's graph + safetensors + golden. Returns safetensors sha256."""
    model = seeded_model(task)
    model2 = seeded_model(task)
    sd, sd2 = model.state_dict(), model2.state_dict()
    assert sorted(sd) == sorted(sd2)
    for k in sd:
        assert torch.equal(sd[k], sd2[k]), f"nondeterministic weight {k}"

    st_path = out / f"model_{task}.safetensors"
    digest = safetensors_bytes_digest(model, st_path)
    tmp = out / f"model_{task}.safetensors.recheck"
    digest2 = safetensors_bytes_digest(model2, tmp)
    tmp.unlink()
    assert digest == digest2, f"safetensors bytes not deterministic: {digest} != {digest2}"

    cfg = x_configs.fixture()
    graph_path = out / f"graph_tabicl_{task}.onnx"
    wrapper = x_export.export_graph(model, graph_path, dim_rows=cfg.dim_rows,
                                    dim_train=cfg.dim_train, dim_features=cfg.dim_features,
                                    example=cfg.example)
    tensor_map = x_export.postprocess(graph_path, dict(model.state_dict()))
    x_export.write_tensor_map(out / f"tensor_map_tabicl_{task}.json", tensor_map,
                              task=task, safetensors_rel=f"model_{task}.safetensors")

    parity = x_export.check_parity(graph_path, wrapper, cfg.parity_shapes)
    assert parity["ok"], parity

    x_export.delete_weight_data(graph_path)
    x_export.assert_weight_free(graph_path, tensor_map)

    x, y = golden_inputs(task)
    with torch.no_grad():
        logits = wrapper(x, y)
    if task == "classification":
        out_doc = "classification: C = max_classes (class logits)."
    else:
        out_doc = ("regression: C = 1 — a single real-valued POINT ESTIMATE per row "
                   "(mean over the quantile head + in-graph inverse StandardScaler). "
                   "y holds RAW train targets; the graph z-scores them internally, so "
                   "the engine feeds RAW targets and reads the output directly.")
    y_convention = (
        "y holds ONLY the training labels (length S = train_size); train_size is "
        "implicit as len(y). logits are [1, T, C]; the runtime reads predictions "
        "on rows >= S (test rows). " + out_doc)
    (out / f"golden_{task}.json").write_text(json.dumps({
        "_doc": {
            "purpose": f"C++ parity: safetensors -> initializer injection -> ORT run "
                       f"on graph_tabicl_{task}.onnx must reproduce these fp32 logits.",
            "parity_slice": "asserted on logits[:, train_size:, :] (test rows)",
            "rtol": PARITY_RTOL, "y_convention": y_convention,
        },
        "inputs": {"x": x.tolist(), "y": y.tolist(), "train_size": GOLDEN_SHAPE["s"]},
        "logits": logits.tolist(),
        "output_shape": list(logits.shape),
        "safetensors_sha256": digest,
    }, indent=2) + "\n")
    return digest


def build(out: pathlib.Path, task: str | None = None) -> dict:
    """Build the dual-task committed CI fixture (both tasks + combined manifest).

    ``task`` is accepted for CLI back-compat but ignored: the fixture always
    ships both classification and regression.
    """
    apply()
    out.mkdir(parents=True, exist_ok=True)
    for legacy in _LEGACY:
        (out / legacy).unlink(missing_ok=True)

    digests = {t: _build_task(out, t) for t in TASKS}

    def file_entry(name):
        p = out / name
        return {"path": name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}

    manifest = {
        "schema_version": 2, "id": "tabicl-fixture",
        "display_name": "TabICL CI fixture (random init, schema v2)",
        "family": "icl-transformer",
        "license": {"id": "bsd-3-clause", "commercial": True,
                    "redistributable": True, "gate_setting": None},
        "preprocessing_profile": "tabicl_v2_minimal",
        "weights": {
            t: {"repo": "local:test/fixtures/tabicl", "revision": "fixture-v1",
                "files": [file_entry(f"model_{t}.safetensors")]}
            for t in TASKS
        },
        "graph": {"classification": "graph_tabicl_classification.onnx",
                  "regression": "graph_tabicl_regression.onnx",
                  "tensor_map": SHARED_TENSOR_MAP},
        "capabilities": ["classify", "regress"],
        "tensor_contract": {
            "inputs": {
                "features": {"name": "x", "dtype": "f32", "shape": ["1", "T", "H"]},
                "labels":   {"name": "y", "dtype": "f32", "shape": ["1", "S"]},
            },
            "outputs": {"logits": {"name": "logits", "dtype": "f32", "shape": ["1", "T", "C"]}},
        },
        "size_regime": {"max_rows": 4096, "max_features": 64, "max_classes": 3},
        "compute": {"cpu": "f32"},
        "_note": "Random-init TabICL fixture (BSD-3-Clause architecture, our weights). "
                 "H is DYNAMIC; y carries train labels only (length=train_size); no "
                 "cat_mask/d/train_size inputs. classification logits are class logits "
                 "(C=max_classes); regression logits are a SINGLE point estimate "
                 "(C=1) — mean over the quantile head + in-graph inverse StandardScaler "
                 "on RAW train targets. One shared tensor_map serves both graphs "
                 "(name-based injection with an 'm.' fallback).",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    hashes = {name: sha256_file(out / name) for name in FILES}
    total = sum((out / name).stat().st_size for name in FILES)
    assert total < 5 * 1024 * 1024, f"fixture total {total} B >= 5 MB budget"
    return hashes
