"""CI fixture model for TabDPT — random weights, tiny dims, deterministic bytes.

Same shape as the Orion-BiX / TabICL fixtures: a tiny random-init TabDPTModel
(Apache-2.0 architecture, OUR weights — zero Layer 6 checkpoint bytes) with
SEEDED weights over SORTED state_dict keys, weight-free graphs through the SAME
exporter, per-task ``golden_<task>.json`` (PyTorch fp32 logits for the C++ parity
test) and a v2 manifest.

**Both tasks, one trunk.** TabDPT is the first built-in whose single checkpoint
serves classification and regression from ONE head: the output's first ``n_out``
columns are class logits and the rest are regression bins. The two graphs
therefore map the SAME tensors and the fixture writes ONE safetensors that both
tasks share, rather than the per-task weight files the other fixtures carry.

**No key prefix.** Unlike Orion-BiX, the released TabDPT safetensors keys are
already the bare module namespace (see ``export.CKPT_KEY_PREFIX``), so the
fixture writes bare keys too and one tensor map serves fixture and real weights.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import torch
from safetensors.torch import save_file

from export_tabdpt import configs as x_configs
from export_tabdpt import export as x_export
from export_tabdpt.cli import build_model
from export_tabdpt.tabdpt_patches import apply, build_wrapper

SEED_WEIGHTS = 1337
SEED_GOLDEN_INPUTS = 7
WEIGHT_SCALE = 0.05
GOLDEN_SHAPE = dict(t=20, h=5, s=12)  # T rows, H features, S train_size
PARITY_RTOL = 1e-4

TASKS = ("classification", "regression")
WEIGHTS_FILE = "model.safetensors"
FILES = [
    "graph_tabdpt_classification.onnx",
    "graph_tabdpt_regression.onnx",
    "tensor_map_tabdpt_classification.json",
    "tensor_map_tabdpt_regression.json",
    "golden_classification.json",
    "golden_regression.json",
    WEIGHTS_FILE,
    "manifest.json",
]


def seeded_model():
    cfg = x_configs.fixture()
    model = build_model(cfg, seed=0)
    gen = torch.Generator().manual_seed(SEED_WEIGHTS)
    with torch.no_grad():
        sd = model.state_dict()
        for key in sorted(sd):  # sorted: init-order independent
            p = sd[key]
            if p.dtype.is_floating_point:
                p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * WEIGHT_SCALE)
    return model.eval()


def safetensors_bytes_digest(model, st_path: pathlib.Path) -> str:
    save_file(
        dict(model.state_dict()), str(st_path),
        metadata={"origin": "anofox-tabfm CI fixture, random init, Apache-2.0 TabDPT "
                            "architecture (our weights, not Layer 6's)"})
    return hashlib.sha256(st_path.read_bytes()).hexdigest()


def golden_inputs(task: str):
    gen = torch.Generator().manual_seed(SEED_GOLDEN_INPUTS)
    t, h, s = (GOLDEN_SHAPE[k] for k in ("t", "h", "s"))
    x = torch.randn(1, t, h, generator=gen)
    if task == "classification":
        y = torch.randint(0, 3, (1, s), generator=gen).float()
    else:
        y = torch.randn(1, s, generator=gen)
    return x, y


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(out: pathlib.Path) -> dict:
    """Build the committed CI fixture (graphs + weights + goldens + manifest)."""
    apply()
    out.mkdir(parents=True, exist_ok=True)

    model = seeded_model()
    model2 = seeded_model()
    sd, sd2 = model.state_dict(), model2.state_dict()
    assert sorted(sd) == sorted(sd2)
    for k in sd:
        assert torch.equal(sd[k], sd2[k]), f"nondeterministic weight {k}"

    st_path = out / WEIGHTS_FILE
    digest = safetensors_bytes_digest(model, st_path)
    tmp = out / (WEIGHTS_FILE + ".recheck")
    digest2 = safetensors_bytes_digest(model2, tmp)
    tmp.unlink()
    assert digest == digest2, f"safetensors bytes not deterministic: {digest} != {digest2}"

    cfg = x_configs.fixture()
    for task in TASKS:
        wrapper = build_wrapper(model, task)
        graph_path = out / f"graph_tabdpt_{task}.onnx"
        x_export.export_graph(wrapper, graph_path, cfg=cfg)
        tensor_map = x_export.postprocess(graph_path, dict(model.state_dict()))
        x_export.write_tensor_map(out / f"tensor_map_tabdpt_{task}.json", tensor_map,
                                  task=task, safetensors_rel=WEIGHTS_FILE)

        parity = x_export.check_parity(graph_path, wrapper, cfg.parity_shapes)
        assert parity["ok"], parity

        x_export.delete_weight_data(graph_path)
        x_export.assert_weight_free(graph_path, tensor_map)

        x, y = golden_inputs(task)
        with torch.no_grad():
            logits = wrapper(x, y)
        out_desc = ("class logits [1, T, C] with C = max_classes"
                    if task == "classification"
                    else "a RAW-space point estimate [1, T, 1] (bar-distribution mean "
                         "over the regression bins, with the train-prefix target "
                         "standardization undone in-graph)")
        (out / f"golden_{task}.json").write_text(json.dumps({
            "_doc": {
                "purpose": "C++ parity: safetensors -> initializer injection -> ORT run "
                           f"on graph_tabdpt_{task}.onnx must reproduce these fp32 logits.",
                "parity_slice": "asserted on logits[:, train_size:, :] (test rows)",
                "rtol": PARITY_RTOL,
                "y_convention": "y holds ONLY the training labels (length S = train_size); "
                                "train_size is implicit as len(y). The runtime reads "
                                f"predictions on rows >= S (test rows). Output is {out_desc}.",
            },
            "inputs": {"x": x.tolist(), "y": y.tolist(), "train_size": GOLDEN_SHAPE["s"]},
            "logits": logits.tolist(),
            "output_shape": list(logits.shape),
            "safetensors_sha256": digest,
        }, indent=2) + "\n")

    def file_entry(name):
        p = out / name
        return {"path": name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}

    manifest = {
        "schema_version": 2, "id": "tabdpt-fixture",
        "display_name": "TabDPT CI fixture (random init, schema v2)",
        "family": "icl-transformer",
        "license": {"id": "apache-2.0", "commercial": True,
                    "redistributable": True, "gate_setting": None},
        # A "*_raw" profile: TabDPT normalizes features INSIDE the forward
        # (clip_outliers + normalize_data over the train prefix), and the wrapper
        # bakes in the target standardization the estimator does outside, so the
        # graph is raw-in / raw-out like TabPFN's and TabICL's.
        "preprocessing_profile": "tabdpt_v1_raw",
        "weights": {
            task: {"repo": "local:test/fixtures/tabdpt", "revision": "fixture-v1",
                   "files": [file_entry(WEIGHTS_FILE)]}
            for task in TASKS
        },
        "graph": {
            "classification": "graph_tabdpt_classification.onnx",
            "regression": "graph_tabdpt_regression.onnx",
            "tensor_map": {
                "classification": "tensor_map_tabdpt_classification.json",
                "regression": "tensor_map_tabdpt_regression.json",
            },
        },
        "capabilities": ["classify", "regress"],
        "tensor_contract": {
            "inputs": {
                "features": {"name": "x", "dtype": "f32", "shape": ["1", "T", "H"]},
                "labels":   {"name": "y", "dtype": "f32", "shape": ["1", "S"]},
            },
            "outputs": {"logits": {"name": "logits", "dtype": "f32", "shape": ["1", "T", "C"]}},
        },
        "size_regime": {"max_rows": 4096, "max_features": 16, "max_classes": 4},
        "compute": {"cpu": "f32"},
        "_note": "Random-init TabDPT fixture (Apache-2.0 architecture, our weights). H is "
                 "DYNAMIC up to num_features (the wrapper pads to it); y carries train "
                 "labels only (length=train_size); no cat_mask/d/train_size inputs. ONE "
                 "safetensors serves BOTH tasks — TabDPT's single head emits class logits "
                 "and regression bins side by side, so the two graphs map the same tensors.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    hashes = {name: sha256_file(out / name) for name in FILES}
    total = sum((out / name).stat().st_size for name in FILES)
    assert total < 5 * 1024 * 1024, f"fixture total {total} B >= 5 MB budget"
    return hashes
