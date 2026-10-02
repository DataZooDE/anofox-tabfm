"""CI fixture model for Causilo: random weights, tiny dims, deterministic bytes.

Same shape as the TabICL / Mitra fixtures (test/fixtures/tabicl, .../mitra): a tiny
random-init Causilo (Apache-2.0 architecture, OUR weights, zero Nums AI checkpoint bytes,
see CLAUDE.md "license wall"), weight-free graphs through the same exporter, and one v2
manifest carrying both tasks.

Two things are different from those fixtures on purpose.

  * The weights are PyTorch's seeded default initialisation with two scale factors
    (WEIGHT_GAIN, TARGET_GAIN), not every parameter overwritten with tiny noise. Both extremes
    failed, for opposite reasons, and each failure was only visible by measuring:
      - DEFAULT init is nearly input-blind. The classification logits vary by 0.008 across
        rows and move by 0.003 when a training label changes: the model ignores its input, so
        a fixture built on it cannot tell a right answer from a wrong one. (It also made every
        ORT-vs-PyTorch parity check pass vacuously, which is worth knowing about the checks.)
      - Turning EVERY projection up (x6) makes it responsive and ill-conditioned: a 1e-7
        relative input change, rounding noise, moved the outputs by up to 0.96, because sharp
        attention amplifies noise. ORT and PyTorch then legitimately disagree by ~5, which is
        not a graph bug, and no cross-platform golden value is stable.
    What works is to keep attention modest (x2) and turn up only the label-embedding
    projections (x20): the output then depends on the training labels and the features without
    sharp attention. With seed 1337 that predicts all three classes (minimum top-1/top-2 margin
    0.12) while a 1e-6 relative input perturbation moves the output by ~4e-5, i.e. the margin is
    ~3,000x the noise it could meet. `_assert_well_conditioned` keeps it that way.
  * The golden inputs are EXACT in binary floating point and expressible in SQL, so the
    SQL test can build the same table and assert a prediction for EVERY row, context and
    query, instead of only that the output is non-NULL. `golden_<task>.json` carries the
    expected per-row values; `sql_inputs` below is the formula the test repeats.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import torch
from safetensors.torch import save_file

from export_causilo import configs as x_configs
from export_causilo import export as x_export
from export_causilo.causilo_patches import ExportWrapper, apply, build_model

SEED_WEIGHTS = 1337
WEIGHT_GAIN = 2.0                          # attention / feed-forward projections; see the docstring
TARGET_GAIN = 20.0                         # label-embedding projections (feature_target, row_target)
MIN_LABEL_SENSITIVITY = 0.3                # output change when one training label changes
MAX_NOISE_RESPONSE = 1e-3                  # output change per 1e-6 relative input perturbation
GOLDEN_SHAPE = dict(t=20, h=5, s=12)      # T rows, H features, S train rows
PARITY_RTOL = 1e-4
MIN_MARGIN = 0.05                          # top-1 minus top-2 logit, every row: no near-ties

TASKS = ("classification", "regression")
FILES = [
    "graph_causilo_classification.onnx", "model_classification.safetensors",
    "tensor_map_causilo_classification.json", "golden_classification.json",
    "graph_causilo_regression.onnx", "model_regression.safetensors",
    "tensor_map_causilo_regression.json", "golden_regression.json",
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


def sensitive(model):
    """Scale a default-initialised model so it responds to its input without amplifying noise
    (see the module docstring). Used by the fixture AND by the tests, which would otherwise run
    on an input-blind model and pass vacuously."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("frequencies", "latents", "missing")) or param.ndim != 2:
                continue
            param.mul_(TARGET_GAIN if name.startswith(("feature_target", "row_target")) else WEIGHT_GAIN)
    return model


def seeded_model(task: str):
    # build_model seeds torch before constructing, so two calls give identical weights.
    return sensitive(build_model(task, x_configs.fixture(), seed=SEED_WEIGHTS))


def _assert_well_conditioned(wrapper, x, y, task: str) -> None:
    """The fixture must respond to its input and must NOT amplify rounding noise."""
    s = y.shape[1]
    with torch.no_grad():
        base = wrapper(x, y)
        noise = 0.0
        for k in range(3):
            g = torch.Generator().manual_seed(k)
            xp = x * (1 + 1e-6 * torch.randn(x.shape, generator=g))
            noise = max(noise, (wrapper(xp, y) - base).abs().max().item())
        y2 = y.clone()
        y2[0, 0] = (y2[0, 0] + 1) % 3 if task == "classification" else y2[0, 0] + 1.5
        label = (wrapper(x, y2)[0, s:] - base[0, s:]).abs().max().item()
    assert noise < MAX_NOISE_RESPONSE, (
        f"ill-conditioned fixture: a 1e-6 relative input change moves the output by {noise:.2e}; "
        "ORT and PyTorch would disagree by more than rounding and no golden value is stable")
    assert label > MIN_LABEL_SENSITIVITY / 3, (
        f"input-blind fixture: changing a training label moves the query outputs by only {label:.2e}")


def safetensors_digest(model, path: pathlib.Path) -> str:
    sd = {k: v.detach().contiguous() for k, v in sorted(model.state_dict().items())}
    save_file(sd, str(path),
              metadata={"origin": "anofox-tabfm CI fixture, random init, Apache-2.0 Causilo "
                                  "architecture (our weights, not Nums AI's)"})
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

    cfg = x_configs.fixture()
    classes = cfg.classes if task == "classification" else 0
    graph_path = out / f"graph_causilo_{task}.onnx"
    wrapper = x_export.export_graph(model, graph_path, example=cfg.example, classes=classes)
    tensor_map = x_export.postprocess(graph_path, dict(model.state_dict()))
    x_export.write_tensor_map(out / f"tensor_map_causilo_{task}.json", tensor_map,
                              task=task, safetensors_rel=f"model_{task}.safetensors")
    parity = x_export.check_parity(graph_path, wrapper, cfg.parity_shapes, classes)
    assert parity["ok"], parity
    x_export.delete_weight_data(graph_path)
    x_export.assert_weight_free(graph_path, tensor_map)

    xs, ys = sql_inputs(task)
    x = torch.tensor([xs], dtype=torch.float32)
    y = torch.tensor([ys], dtype=torch.float32)
    _assert_well_conditioned(wrapper, x, y, task)
    with torch.no_grad():
        logits = wrapper(x, y)                                     # [1, T, C]
    s = GOLDEN_SHAPE["s"]
    per_row = {}
    if task == "classification":
        top2 = logits[0].topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).min().item()
        assert margin > MIN_MARGIN, (
            f"a golden row is a near-tie (min margin {margin:.3g}); the SQL test would be "
            "flaky across platforms. Change the seed or the input formulas.")
        per_row["labels"] = [f"c{int(c)}" for c in logits[0].argmax(-1).tolist()]
    else:
        values = logits[0, :, 0]
        assert values.std().item() > 0.05, "golden regression output is nearly constant"
        per_row["values"] = [round(v, 6) for v in values.tolist()]
    assert logits[0].std(dim=0).max().item() > 0.05, "golden output is nearly constant across rows"

    (out / f"golden_{task}.json").write_text(json.dumps({
        "_doc": {
            "purpose": f"Parity: safetensors -> initializer injection -> ORT on "
                       f"graph_causilo_{task}.onnx must reproduce these fp32 outputs, for EVERY "
                       f"row (rows < S are in-context fitted values, rows >= S are predictions).",
            "rtol": PARITY_RTOL,
            "y_convention": "y holds ONLY the training targets (length S); S = len(y).",
            "inputs": "x[i][j] = ((i*7 + j*3 + (i*j) % 5) % 11) * 0.25 - 1.0; "
                      + ("y[i] = (i*5 + i//3) % 3" if task == "classification"
                         else "y[i] = ((i*37) % 17) * 0.25 - 2.0"),
        },
        "inputs": {"x": x.tolist(), "y": y.tolist(), "train_size": s},
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
        "schema_version": 2, "id": "causilo-fixture",
        "display_name": "Causilo CI fixture (random init, schema v2)",
        "family": "icl-transformer",
        "license": {"id": "apache-2.0", "commercial": True, "redistributable": True,
                    "gate_setting": None},
        "preprocessing_profile": "causilo_v1_raw",
        "weights": {
            t: {"repo": "local:test/fixtures/causilo", "revision": "fixture-v1",
                "files": [file_entry(f"model_{t}.safetensors")]}
            for t in TASKS
        },
        "graph": {"classification": "graph_causilo_classification.onnx",
                  "regression": "graph_causilo_regression.onnx",
                  "tensor_map": {"classification": "tensor_map_causilo_classification.json",
                                 "regression": "tensor_map_causilo_regression.json"}},
        "capabilities": ["classify", "regress"],
        "tensor_contract": {
            "inputs": {
                "features": {"name": "x", "dtype": "f32", "shape": ["1", "T", "H"]},
                "labels": {"name": "y", "dtype": "f32", "shape": ["1", "S"]},
            },
            "outputs": {"logits": {"name": "logits", "dtype": "f32", "shape": ["1", "T", "C"]}},
        },
        "size_regime": {"max_rows": 4096, "max_features": 64, "max_classes": 3},
        "compute": {"cpu": "f32"},
        "_note": "Random-init Causilo fixture (Apache-2.0 architecture, our weights). H is "
                 "DYNAMIC; y carries train targets only (length = train_size); no "
                 "cat_mask/d/train_size inputs. The graph normalises features in-graph, so "
                 "the engine profile is *_raw. classification outputs are class logits "
                 "(C = max_classes); regression outputs are a SINGLE RAW-space point estimate "
                 "(C = 1). Rows < S are in-context fitted values (context rows presented a "
                 "second time as queries).",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    hashes = {name: sha256_file(out / name) for name in FILES}
    total = sum((out / name).stat().st_size for name in FILES)
    assert total < 5 * 1024 * 1024, f"fixture total {total} B >= 5 MB budget"
    return hashes
