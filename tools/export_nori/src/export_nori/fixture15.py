"""CI fixture model for Nori 1.5 (Model v3, two-phase encoder): random weights, tiny dims, deterministic bytes.

Same idea as ``fixture.py`` (the Nori-30M fixture) and it reuses that module's golden inputs and conditioning
guard: a tiny random-init model (Apache-2.0 architecture, OUR weights, zero Synthefy checkpoint bytes: see
CLAUDE.md "license wall") at every structural flag of the released config, a weight-free graph through the
same exporter, and a v2 manifest with the one task the model has.

Differences from the 30M fixture:

  * the weights file is a ``torch.save`` checkpoint in the RELEASED Model v3 layout,
    ``{"model_config", "model_state_dict", "inference_config"}`` (no EMA copy), so the fixture exercises the
    native reader's ``model_state_dict`` path, which the 30M fixture's EMA layout cannot;
  * the gains differ. A default-initialised two-phase stack is nearly input-blind just like the 30M one (the
    attention output and SwiGLU down projections are initialised at std 0.02 under a DeepNorm residual scale), so
    the same three levers are pulled: the target-aware embedding (TARGET_GAIN), the attention-output / SwiGLU-down
    projections (MIX_GAIN) and one shared direction plus an ordering ramp in the quantile head (HEAD_GAIN). They
    were found by ``tools/export_nori`` sweeping seeds 0-3 against ``fixture.check_well_conditioned``; seed 0
    passes at (10, 5, 20) with an output std of 0.87 over the 20 golden rows.

The golden inputs are exact in binary floating point and expressible in SQL, so the SQL test builds the same
table and pins a prediction for EVERY row, context and query.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import torch

from export_nori import configs as x_configs
from export_nori import export as x_export
from export_nori import fixture as base
from export_nori.nori_patches import build_model

SEED_WEIGHTS = 0
TARGET_GAIN = 10.0
MIX_GAIN = 5.0
HEAD_GAIN = 20.0
GOLDEN_SHAPE = base.GOLDEN_SHAPE
TASK = "regression"
WEIGHTS = "model.ckpt"
FILES = ["graph_nori_regression.onnx", "tensor_map_nori_regression.json", "golden_regression.json",
         WEIGHTS, "manifest.json"]
INFERENCE_CONFIG = {"name": "model_v3_inference.json", "members": []}


def sensitive(model):
    """Scale ONLY the target-side projections of a default-initialised model (see the docstring)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim < 2:
                continue
            if "reg_target_aware_embedding" in name:
                param.mul_(TARGET_GAIN)
            elif name.endswith(("out_proj_weight", "w_down.weight")):
                param.mul_(MIX_GAIN)
        head = model.reg_y_decoder[-1].weight
        k = head.shape[0]
        ramp = (torch.arange(k, dtype=head.dtype) - k / 2).unsqueeze(1) / k
        head.copy_(HEAD_GAIN * (head[:1] + ramp * head[1:2]))
    return model


def seeded_model(seed: int = SEED_WEIGHTS):
    # build_model seeds torch before constructing, so two calls give identical weights.
    return sensitive(build_model(x_configs.fixture15().model_config, seed=seed)).eval()


def save_checkpoint(model, path: pathlib.Path) -> str:
    """The released file's layout: {"model_config", "model_state_dict", "inference_config"}.
    torch.save output is deterministic for the same tensors and config (checked by ``build``)."""
    sd = {k: v.detach().contiguous().clone() for k, v in sorted(model.state_dict().items())}
    torch.save({"model_config": x_configs.fixture15().model_config, "model_state_dict": sd,
                "inference_config": INFERENCE_CONFIG}, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: pathlib.Path) -> dict:
    """Build the committed CI fixture (regression graph + checkpoint + golden + manifest)."""
    out.mkdir(parents=True, exist_ok=True)
    model, model2 = seeded_model(), seeded_model()
    for k, v in model.state_dict().items():
        assert torch.equal(v, model2.state_dict()[k]), f"nondeterministic weight {k}"
    # torch.save names the archive root after the file stem, so the recheck must have the same stem.
    digest = save_checkpoint(model, out / WEIGHTS)
    recheck_dir = out / ".recheck"
    recheck_dir.mkdir(exist_ok=True)
    digest2 = save_checkpoint(model2, recheck_dir / WEIGHTS)
    (recheck_dir / WEIGHTS).unlink()
    recheck_dir.rmdir()
    assert digest == digest2, f"checkpoint bytes not deterministic: {digest} != {digest2}"

    cfg = x_configs.fixture15()
    graph_path = out / f"graph_nori_{TASK}.onnx"
    wrapper = x_export.export_graph(model, graph_path, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                                    dim_features=cfg.dim_features, example=cfg.example)
    tensor_map = x_export.postprocess(graph_path, dict(model.state_dict()))
    x_export.write_tensor_map(out / f"tensor_map_nori_{TASK}.json", tensor_map, task=TASK,
                              safetensors_rel=WEIGHTS)
    parity = x_export.check_parity(graph_path, wrapper, cfg.parity_shapes, tol=1e-3)
    assert parity["ok"], parity
    x_export.delete_weight_data(graph_path)
    x_export.assert_weight_free(graph_path, tensor_map)

    x, y = base.golden_tensors()
    base.assert_well_conditioned(wrapper, model, x, y)
    with torch.no_grad():
        out_t = wrapper(x, y)                                         # [1, T, 1]
    s = GOLDEN_SHAPE["s"]
    values = [round(v, 6) for v in out_t[0, :, 0].tolist()]
    (out / f"golden_{TASK}.json").write_text(json.dumps({
        "_doc": {
            "purpose": "Parity: checkpoint -> initializer injection -> ORT on graph_nori_regression.onnx must "
                       "reproduce these fp32 outputs, for EVERY row (rows < S are in-context fitted values, "
                       "rows >= S are predictions).",
            "y_convention": "y holds ONLY the RAW training targets (length S); S = len(y).",
            "inputs": "x[i][j] = ((i*7 + j*3 + (i*j) % 5) % 11) * 0.25 - 1.0 + "
                      "((i*13 + j*5 + (i*j) % 3) % 7) * 0.0625; y[i] = ((i*37) % 17) * 0.25 - 2.0",
        },
        "inputs": {"x": x.tolist(), "y": y.tolist(), "train_size": s},
        "outputs": out_t.tolist(),
        "output_shape": list(out_t.shape),
        "per_row": {"values": values},
        "checkpoint_sha256": digest,
    }, indent=2) + "\n")

    def file_entry(name):
        p = out / name
        return {"path": name, "bytes": p.stat().st_size, "sha256": base.sha256_file(p)}

    manifest = {
        "schema_version": 2, "id": "nori15-fixture",
        "display_name": "Nori 1.5 CI fixture (random init, schema v2)",
        "family": "icl-transformer",
        "license": {"id": "apache-2.0", "commercial": True, "redistributable": True, "gate_setting": None},
        "preprocessing_profile": "nori_v15_raw",
        "weights": {TASK: {"repo": "local:test/fixtures/nori15", "revision": "fixture-v1",
                           "files": [file_entry(WEIGHTS)]}},
        "graph": {TASK: f"graph_nori_{TASK}.onnx", "tensor_map": {TASK: f"tensor_map_nori_{TASK}.json"}},
        "capabilities": ["regress"],
        "tensor_contract": {
            "inputs": {
                "features": {"name": "x", "dtype": "f32", "shape": ["1", "T", "H"]},
                "labels": {"name": "y", "dtype": "f32", "shape": ["1", "S"]},
            },
            "outputs": {"logits": {"name": "logits", "dtype": "f32", "shape": ["1", "T", "1"]}},
        },
        "size_regime": {"max_rows": 4096, "max_features": 64, "max_classes": 10},
        "compute": {"cpu": "f32"},
        "_note": "Random-init Nori 1.5 (Model v3, two-phase encoder) fixture (Apache-2.0 architecture, our "
                 "weights), regression only, in the released checkpoint's own torch.save layout "
                 "{model_config, model_state_dict, inference_config} (no EMA copy). H is DYNAMIC; y carries RAW "
                 "train targets only (length = train_size). The model standardises features itself and the "
                 "wrapper standardises the target in-graph, so the engine profile is *_raw. The output is ONE "
                 "raw-space point estimate per row (the mean of the quantile head); rows < S are in-context "
                 "fitted values (context rows presented a second time as queries).",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    hashes = {name: base.sha256_file(out / name) for name in FILES}
    total = sum((out / name).stat().st_size for name in FILES)
    assert total < 5 * 1024 * 1024, f"fixture total {total} B >= 5 MB budget"
    return hashes
