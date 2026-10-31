"""CI fixture model for EXAONE Tabular: random weights, tiny dims, deterministic bytes.

Same shape as the Causilo fixture: a tiny random-init EXAONE Tabular (BSD-3-Clause-LG architecture, OUR
weights, zero LG AI Research checkpoint bytes: see CLAUDE.md "license wall") at every structural flag of
the released configs, weight-free graphs through the same exporter, and one v2 manifest carrying both
tasks. Unlike LimiX/Nori there are TWO weights files, one per task, because LG release the classifier and
the regressor as separate checkpoints (different heads, different label encoders).

The golden inputs are exact in binary floating point and expressible in SQL, so the SQL test builds the
same table and pins a prediction for EVERY row, context and query. The builder asserts the fixture
responds to its labels, is not ill-conditioned (1e-6 relative input noise must not move the output), has
no near-tie between the top two classes, and does not collapse to one class (context rows included).

A default initialisation is nearly input-blind, so ``sensitive`` scales a few projections (GAINS):
the label encoder (so queries depend on the training labels), and the head's output layer, after
removing what makes every row decode alike. For the quantile head that is K independent random rows
averaging out in the mean point estimate (they get one shared direction plus an ordering ramp, as a
trained head's quantiles share a location); for the class head it is the positive mean of the GELU
hidden layer, which lets each random row's SUM pick the class for every input (each row is centred).
Even so a random 2-block CAST predicts two of the three classes for most seeds and all three for none
of the 160 seed/gain combinations tried, so the class check asks for two (MIN_CLASSES_PREDICTED).
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import torch
from safetensors.torch import save_file

from export_exaone import configs as x_configs
from export_exaone import export as x_export
from export_exaone.exaone_patches import ExportWrapper, build_model

SEED_WEIGHTS = {"classification": 6, "regression": 0}
# (label encoder, every attention output projection, the head's output layer) gains; see sensitive()
GAINS = {"classification": (1.0, 1.0, 10.0), "regression": (4.0, 1.0, 10.0)}
MIN_LABEL_SENSITIVITY = 0.1
MAX_NOISE_RESPONSE = 1e-2
MIN_MARGIN = 0.05
MIN_CLASSES_PREDICTED = 2                  # of 3; see the module docstring
MIN_SPREAD = 0.05
GOLDEN_SHAPE = dict(t=20, h=5, s=12)
TASKS = ("classification", "regression")
FILES = [
    "graph_exaone_classification.onnx", "tensor_map_exaone_classification.json", "golden_classification.json",
    "graph_exaone_regression.onnx", "tensor_map_exaone_regression.json", "golden_regression.json",
    "model_classification.safetensors", "model_regression.safetensors", "manifest.json",
]


def sql_inputs(task: str):
    """(x [T,H], y [S]) as nested lists, from formulas the SQL test repeats verbatim.

    x[i][j] = ((i*7 + j*3 + (i*j) % 5) % 11) * 0.25 - 1.0 + ((i*13 + j*5 + (i*j) % 3) % 7) * 0.0625
    classification y[i] = (i*5 + i//3) % 3                     labels 'c0','c1','c2'
    regression     y[i] = ((i*37) % 17) * 0.25 - 2.0           exact in float32
    """
    t, h, s = (GOLDEN_SHAPE[k] for k in ("t", "h", "s"))
    x = [[((i * 7 + j * 3 + (i * j) % 5) % 11) * 0.25 - 1.0 + ((i * 13 + j * 5 + (i * j) % 3) % 7) * 0.0625
          for j in range(h)] for i in range(t)]
    if task == "classification":
        y = [float((i * 5 + i // 3) % 3) for i in range(s)]
    else:
        y = [((i * 37) % 17) * 0.25 - 2.0 for i in range(s)]
    return x, y


def golden_tensors(task: str):
    xs, ys = sql_inputs(task)
    return torch.tensor([xs], dtype=torch.float32), torch.tensor([ys], dtype=torch.float32)


def sensitive(model, task: str):
    """Make a random model respond to its inputs (see the module docstring for what each gain is for)."""
    label_gain, mix_gain, head_gain = GAINS[task]
    with torch.no_grad():
        model.label_encoder.weight.mul_(label_gain)
        for name, param in model.named_parameters():
            if name.endswith("output_weight"):
                param.mul_(mix_gain)
        head = model.transformer.classification_heads["standard"][-1]
        if task == "regression":
            # quantile head: K independent random rows average out in the mean point estimate; give them one
            # shared direction plus an ordering ramp, as a trained head's quantiles share a location
            k = head.weight.shape[0]
            ramp = (torch.arange(k, dtype=head.weight.dtype) - k / 2).unsqueeze(1) / k
            head.weight.copy_(head.weight[:1] + ramp * head.weight[1:2])
        else:
            # class head: the GELU hidden layer is mostly positive, so a random output row's SUM decides the
            # class for every input; centring each row (and zeroing the bias) leaves only the input-dependent
            # part, so the fixture predicts more than one class
            head.weight.sub_(head.weight.mean(dim=1, keepdim=True))
            head.bias.zero_()
        head.weight.mul_(head_gain)
    return model


def seeded_model(task: str, seed: int | None = None):
    cfg = x_configs.get("fixture", task)
    seed = SEED_WEIGHTS[task] if seed is None else seed
    return sensitive(build_model(task, cfg.model_config, cfg.regression_config, seed=seed), task).eval()


def check(wrapper, x, y, task: str) -> list:
    """Every reason the fixture is unusable (empty when it is fine)."""
    problems = []
    s = y.shape[1]
    with torch.no_grad():
        base = wrapper(x, y)
        noise = 0.0
        for k in range(3):
            g = torch.Generator().manual_seed(k)
            noise = max(noise, (wrapper(x * (1 + 1e-6 * torch.randn(x.shape, generator=g)), y) - base)
                        .abs().max().item())
        y2 = y.clone()
        y2[0, 0] = (y2[0, 0] + 1) % 3 if task == "classification" else y2[0, 0] + 1.5
        sens = (wrapper(x, y2)[0, s:] - base[0, s:]).abs().max().item()
    if noise >= MAX_NOISE_RESPONSE:
        problems.append(f"ill-conditioned: 1e-6 relative input noise moves the output by {noise:.2e}")
    if sens <= MIN_LABEL_SENSITIVITY:
        problems.append(f"label-blind: changing one training label moves the queries by {sens:.2e}")
    if task == "classification":
        logits = base[0, :, :3]
        top2 = logits.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).min().item()
        if margin <= MIN_MARGIN:
            problems.append(f"near-tie: min top-1/top-2 margin {margin:.3g}")
        labels = logits.argmax(-1)
        if len(set(labels.tolist())) < MIN_CLASSES_PREDICTED:
            problems.append(f"predicts fewer than {MIN_CLASSES_PREDICTED} classes")
        if len(set(labels[:s].tolist())) < 2:
            problems.append("the context rows predict a single class")
    else:
        values = base[0, :, 0]
        if values.std().item() <= MIN_SPREAD or values[:s].std().item() <= MIN_SPREAD:
            problems.append(f"nearly constant output (std {values.std().item():.3g})")
    return problems


def safetensors_digest(model, path: pathlib.Path) -> str:
    sd = {k: v.detach().contiguous() for k, v in sorted(model.state_dict().items())}
    save_file(sd, str(path), metadata={"origin": "anofox-tabfm CI fixture, random init, BSD-3-Clause-LG "
                                                 "EXAONE Tabular architecture (our weights, not LG's)"})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _build_task(out: pathlib.Path, task: str) -> str:
    cfg = x_configs.get("fixture", task)
    model, model2 = seeded_model(task), seeded_model(task)
    weights = out / f"model_{task}.safetensors"
    digest = safetensors_digest(model, weights)
    recheck = out / f"model_{task}.safetensors.recheck"
    assert safetensors_digest(model2, recheck) == digest, "safetensors bytes are not deterministic"
    recheck.unlink()

    graph_path = out / f"graph_exaone_{task}.onnx"
    wrapper = x_export.export_graph(model, graph_path, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                                    dim_features=cfg.dim_features, example=cfg.example, task=task)
    tensor_map = x_export.postprocess(graph_path, dict(model.state_dict()))
    x_export.write_tensor_map(out / f"tensor_map_exaone_{task}.json", tensor_map, task=task,
                              safetensors_rel=weights.name)
    parity = x_export.check_parity(graph_path, wrapper, cfg.parity_shapes, task=task, tol=1e-3)
    assert parity["ok"], parity
    x_export.delete_weight_data(graph_path)
    x_export.assert_weight_free(graph_path, tensor_map)

    x, y = golden_tensors(task)
    problems = check(wrapper, x, y, task)
    assert not problems, "; ".join(problems)
    with torch.no_grad():
        logits = wrapper(x, y)
    if task == "classification":
        per_row = {"labels": [f"c{int(c)}" for c in logits[0, :, :3].argmax(-1).tolist()]}
    else:
        per_row = {"values": [round(v, 6) for v in logits[0, :, 0].tolist()]}
    (out / f"golden_{task}.json").write_text(json.dumps({
        "_doc": {
            "purpose": f"Parity: safetensors -> initializer injection -> ORT on graph_exaone_{task}.onnx must "
                       "reproduce these fp32 outputs, for EVERY row (rows < S are in-context fitted values, "
                       "rows >= S are predictions).",
            "y_convention": "y holds ONLY the training labels (length S); S = len(y).",
            "inputs": "x[i][j] = ((i*7 + j*3 + (i*j) % 5) % 11) * 0.25 - 1.0 + "
                      "((i*13 + j*5 + (i*j) % 3) % 7) * 0.0625; "
                      + ("y[i] = (i*5 + i//3) % 3" if task == "classification"
                         else "y[i] = ((i*37) % 17) * 0.25 - 2.0"),
        },
        "inputs": {"x": x.tolist(), "y": y.tolist(), "train_size": GOLDEN_SHAPE["s"]},
        "outputs": logits.tolist(),
        "output_shape": list(logits.shape),
        "per_row": per_row,
        "safetensors_sha256": digest,
    }, indent=2) + "\n")
    return digest


def build(out: pathlib.Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        _build_task(out, task)

    def file_entry(name):
        p = out / name
        return {"path": name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}

    manifest = {
        "schema_version": 2, "id": "exaone-fixture",
        "display_name": "EXAONE Tabular CI fixture (random init, schema v2)",
        "family": "icl-transformer",
        "license": {"id": "bsd-3-clause-lg", "commercial": True, "redistributable": True, "gate_setting": None},
        "preprocessing_profile": "exaone_v1_raw",
        "weights": {
            t: {"repo": "local:test/fixtures/exaone", "revision": "fixture-v1",
                "files": [file_entry(f"model_{t}.safetensors")]}
            for t in TASKS
        },
        "graph": {"classification": "graph_exaone_classification.onnx",
                  "regression": "graph_exaone_regression.onnx",
                  "tensor_map": {"classification": "tensor_map_exaone_classification.json",
                                 "regression": "tensor_map_exaone_regression.json"}},
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
        "_note": "Random-init EXAONE Tabular fixture (BSD-3-Clause-LG architecture, our weights; the fixture "
                 "weights are ours, so the commercial/redistributable flags describe THEM, not LG's released "
                 "checkpoints). One safetensors per task. H is DYNAMIC; y carries train labels only (length = "
                 "train_size). The model standardises its features itself and the wrapper standardises the "
                 "regression target in-graph, so the engine profile is *_raw. Classification outputs class "
                 "logits; regression outputs ONE raw-space point estimate (the mean of the quantile head). Rows "
                 "< S are in-context fitted values (context rows presented a second time as queries).",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    hashes = {name: sha256_file(out / name) for name in FILES}
    total = sum((out / name).stat().st_size for name in FILES)
    assert total < 5 * 1024 * 1024, f"fixture total {total} B >= 5 MB budget"
    return hashes
