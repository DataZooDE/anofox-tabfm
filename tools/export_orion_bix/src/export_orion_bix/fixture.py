"""CI fixture model for Orion-BiX — random weights, tiny dims, deterministic bytes.

Same shape as the TabICL fixture (test/fixtures/tabicl): a tiny random-init
OrionBix (MIT architecture, OUR weights — zero Lexsi checkpoint bytes) with
SEEDED weights over SORTED state_dict keys, a weight-free graph through the SAME
exporter, a ``golden_classification.json`` (PyTorch fp32 logits for the C++
parity test) and a v2 manifest.

**Single capability.** Orion-BiX ships no regression head, so the manifest
declares ``capabilities: ["classify"]`` — this is the first built-in that does
not support both tasks, and `test/sql/tabfm_orion_bix.test` asserts that
`tabfm_regress` against it raises the actionable unsupported-task error.

The safetensors is written with the released checkpoint's ``_orig_mod.`` key
prefix (see ``export.CKPT_KEY_PREFIX``) so the committed tensor map resolves
identically against the fixture and against a real downloaded .ckpt.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import torch
from safetensors.torch import save_file

from export_orion_bix import configs as x_configs
from export_orion_bix import export as x_export
from export_orion_bix.orion_bix_patches import apply, build_model

SEED_WEIGHTS = 1337
SEED_GOLDEN_INPUTS = 7
WEIGHT_SCALE = 0.05
GOLDEN_SHAPE = dict(t=20, h=5, s=12)  # T rows, H features, S train_size
PARITY_RTOL = 1e-4

TASK = "classification"
FILES = [
    "graph_orion_bix_classification.onnx",
    "model_classification.safetensors",
    "tensor_map_orion_bix_classification.json",
    "golden_classification.json",
    "manifest.json",
]


def _prefixed(state_dict: dict) -> dict:
    """Re-key a bare state_dict into the released checkpoint's namespace."""
    return {x_export.CKPT_KEY_PREFIX + k: v for k, v in state_dict.items()}


def seeded_model():
    cfg = x_configs.fixture()
    model = build_model(cfg.model_kwargs, seed=0)
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
        _prefixed(model.state_dict()), str(st_path),
        metadata={"origin": "anofox-tabfm CI fixture, random init, MIT Orion-BiX "
                            "architecture (our weights, not Lexsi's)"})
    return hashlib.sha256(st_path.read_bytes()).hexdigest()


def golden_inputs():
    gen = torch.Generator().manual_seed(SEED_GOLDEN_INPUTS)
    t, h, s = (GOLDEN_SHAPE[k] for k in ("t", "h", "s"))
    x = torch.randn(1, t, h, generator=gen)
    y = torch.randint(0, 3, (1, s), generator=gen).float()
    return x, y


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(out: pathlib.Path) -> dict:
    """Build the committed CI fixture (graph + weights + golden + manifest)."""
    apply()
    out.mkdir(parents=True, exist_ok=True)

    model = seeded_model()
    model2 = seeded_model()
    sd, sd2 = model.state_dict(), model2.state_dict()
    assert sorted(sd) == sorted(sd2)
    for k in sd:
        assert torch.equal(sd[k], sd2[k]), f"nondeterministic weight {k}"

    st_path = out / f"model_{TASK}.safetensors"
    digest = safetensors_bytes_digest(model, st_path)
    tmp = out / f"model_{TASK}.safetensors.recheck"
    digest2 = safetensors_bytes_digest(model2, tmp)
    tmp.unlink()
    assert digest == digest2, f"safetensors bytes not deterministic: {digest} != {digest2}"

    cfg = x_configs.fixture()
    graph_path = out / f"graph_orion_bix_{TASK}.onnx"
    wrapper = x_export.export_graph(model, graph_path, dim_rows=cfg.dim_rows,
                                    dim_train=cfg.dim_train, dim_features=cfg.dim_features,
                                    example=cfg.example)
    tensor_map = x_export.postprocess(graph_path, dict(model.state_dict()))
    x_export.write_tensor_map(out / f"tensor_map_orion_bix_{TASK}.json", tensor_map,
                              task=TASK, safetensors_rel=f"model_{TASK}.safetensors")

    parity = x_export.check_parity(graph_path, wrapper, cfg.parity_shapes)
    assert parity["ok"], parity

    x_export.delete_weight_data(graph_path)
    x_export.assert_weight_free(graph_path, tensor_map)

    x, y = golden_inputs()
    with torch.no_grad():
        logits = wrapper(x, y)
    (out / f"golden_{TASK}.json").write_text(json.dumps({
        "_doc": {
            "purpose": "C++ parity: safetensors -> initializer injection -> ORT run on "
                       f"graph_orion_bix_{TASK}.onnx must reproduce these fp32 logits.",
            "parity_slice": "asserted on logits[:, train_size:, :] (test rows)",
            "rtol": PARITY_RTOL,
            "y_convention": "y holds ONLY the training labels (length S = train_size); "
                            "train_size is implicit as len(y). logits are [1, T, C] class "
                            "logits with C = max_classes; the runtime reads predictions on "
                            "rows >= S (test rows).",
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
        "schema_version": 2, "id": "orion-bix-fixture",
        "display_name": "Orion-BiX CI fixture (random init, schema v2)",
        "family": "icl-transformer",
        "license": {"id": "mit", "commercial": True,
                    "redistributable": True, "gate_setting": None},
        # NOT a "*_raw" profile: unlike TabPFN/TabICL, Orion-BiX does no internal
        # normalization — its sklearn wrapper standardizes externally
        # (CustomStandardScaler / RTDLQuantileTransformer), so the ENGINE must
        # standardize features, exactly as for tabfm-v1 and mitra.
        "preprocessing_profile": "orion_bix_v1_minimal",
        "weights": {
            TASK: {"repo": "local:test/fixtures/orion_bix", "revision": "fixture-v1",
                   "files": [file_entry(f"model_{TASK}.safetensors")]},
        },
        "graph": {TASK: f"graph_orion_bix_{TASK}.onnx",
                  "tensor_map": f"tensor_map_orion_bix_{TASK}.json"},
        "capabilities": ["classify"],
        "tensor_contract": {
            "inputs": {
                "features": {"name": "x", "dtype": "f32", "shape": ["1", "T", "H"]},
                "labels":   {"name": "y", "dtype": "f32", "shape": ["1", "S"]},
            },
            "outputs": {"logits": {"name": "logits", "dtype": "f32", "shape": ["1", "T", "C"]}},
        },
        "size_regime": {"max_rows": 4096, "max_features": 64, "max_classes": 3},
        "compute": {"cpu": "f32"},
        "_note": "Random-init Orion-BiX fixture (MIT architecture, our weights). H is "
                 "DYNAMIC; y carries train labels only (length=train_size); no "
                 "cat_mask/d/train_size inputs. CLASSIFICATION ONLY — upstream ships no "
                 "regression head. safetensors keys carry the released checkpoint's "
                 "'_orig_mod.' prefix so one tensor map serves fixture and real weights.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    hashes = {name: sha256_file(out / name) for name in FILES}
    total = sum((out / name).stat().st_size for name in FILES)
    assert total < 5 * 1024 * 1024, f"fixture total {total} B >= 5 MB budget"
    return hashes
