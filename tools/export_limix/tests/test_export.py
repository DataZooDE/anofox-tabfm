"""LimiX export: is the traced graph right at shapes it was NOT traced at?

The spike's README documents exactly where the first export failed. Those three shapes are the
red test: each is a (T, H, S) at which ONNX Runtime must agree with PyTorch, traced at one
shape (T=20, H=6, S=12). A graph that only works at its trace shape bakes a Python value in.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from export_limix import configs, export
from export_limix.limix_patches import build_model

TRACE = (20, 6, 12)                      # (T, H, S): the shape the README traces at
#: The README's three shapes that failed, verbatim: (T, H, S).
README_FAILING = [(20, 6, 15), (30, 6, 12), (40, 8, 30)]


def _export(tmp_path, task):
    cfg = configs.fixture()
    model = build_model(cfg.model_config, seed=0)
    path = tmp_path / f"g_{task}.onnx"
    wrapper = export.export_graph(model, path, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                                  dim_features=cfg.dim_features, example=TRACE,
                                  max_classes=cfg.max_classes, task=task)
    return cfg, path, wrapper


@pytest.fixture(scope="module", params=["classification", "regression"])
def exported(request, tmp_path_factory):
    return (request.param, *_export(tmp_path_factory.mktemp(request.param), request.param))


@pytest.mark.parametrize("t,h,s", README_FAILING)
def test_graph_runs_and_matches_pytorch_at_a_shape_it_was_not_traced_at(exported, t, h, s):
    task, cfg, path, wrapper = exported
    r = export.check_parity(path, wrapper, ((t, h, s),), max_classes=cfg.max_classes, task=task)
    assert r["ok"], r


#: The README's shapes all have an EVEN H. LimiX groups features in twos and pads with
#: `num_features % features_per_group`, a Python modulo on a symbolic size: traced at H=6 it can only
#: record "no padding needed". Odd widths are the common case in practice.
ODD_WIDTHS = [(20, 5, 12), (24, 7, 10), (33, 9, 21)]


@pytest.mark.parametrize("t,h,s", ODD_WIDTHS)
def test_odd_feature_widths_need_the_group_padding(exported, t, h, s):
    task, cfg, path, wrapper = exported
    r = export.check_parity(path, wrapper, ((t, h, s),), max_classes=cfg.max_classes, task=task)
    assert r["ok"], r


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_replacement_y_decoder_equals_upstreams_on_a_uniform_y_type(task):
    """Patch 5 claims the split is the identity when y_type is uniform. Prove it, don't assert it."""
    cfg = configs.fixture()
    model = build_model(cfg.model_config, seed=3)
    model._export_task = "cls" if task == "classification" else "reg"
    g = torch.Generator().manual_seed(0)
    out = torch.randn(1, 9, cfg.model_config["embed_dim"], generator=g)
    y_type = (torch.zeros if task == "classification" else torch.ones)(1, 9, 1)
    up_cls, up_reg = type(model)._upstream_y_decoder(model, out, y_type)
    new_cls, new_reg = model.y_decoder(out, y_type)
    want, got = (up_cls, new_cls) if task == "classification" else (up_reg, new_reg)
    assert torch.equal(want, got)


# --- the engine decodes EVERY row, so every row must be a real prediction -------------------------

def _wrapper(task, seed=0):
    from export_limix.limix_patches import ExportWrapper
    cfg = configs.fixture()
    return cfg, ExportWrapper(build_model(cfg.model_config, seed=seed), task=task).eval()


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_context_rows_are_not_a_zero_pad(task):
    """The wrapper used to return zeros for rows < S. The engine surfaces those rows as is_training
    fitted values, so every one decoded to the same answer (the defect #51 fixed for other families)."""
    cfg, w = _wrapper(task)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 24, 6, generator=g)
    y = (torch.randint(0, 3, (1, 14), generator=g).float() if task == "classification"
         else torch.randn(1, 14, generator=g) * 3 + 7)
    with torch.no_grad():
        out = w(x, y)
    assert out.shape[1] == 24
    assert out[:, :14].abs().max().item() > 0.0, "context rows are an all-zero pad"


def test_regression_is_raw_in_raw_out():
    """The model never normalises its target: upstream's own examples z-score y before the call and
    invert after. The engine feeds RAW targets and reads the output as raw, so the graph must do both.
    Equivariance is the property: predict(a*y + b) == a*predict(y) + b."""
    cfg, w = _wrapper("regression")
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 20, 6, generator=g)
    y = torch.randn(1, 12, generator=g)
    a, b = 40.0, 150.0
    with torch.no_grad():
        base = w(x, y)
        scaled = w(x, y * a + b)
    assert (scaled - (base * a + b)).abs().max().item() < 1e-3 * (abs(a) * base.abs().max().item() + abs(b))


def test_a_deep_model_exports_in_reasonable_time(tmp_path):
    """Export time must not explode with depth. The real model has 12 layers.

    Without `torch._check(S <= T)` in the wrapper each per-layer slice yields a Min(S, T) that sympy
    re-simplifies in every layer: on the small config 8 layers took more than 480 s (and the ORIGINAL
    code did not finish 4 layers in 15 minutes), against 32 s with it. Run in a subprocess so a regression
    fails in minutes instead of hanging the suite.
    """
    import subprocess
    import sys
    import textwrap
    code = textwrap.dedent("""
        import copy, pathlib, sys
        from export_limix import configs, export
        from export_limix.limix_patches import build_model
        cfg = configs.fixture(); mc = copy.deepcopy(cfg.model_config); mc["nlayers"] = 8
        export.export_graph(build_model(mc, seed=0), pathlib.Path(sys.argv[1]) / "g.onnx", dim_rows=cfg.dim_rows,
                            dim_train=cfg.dim_train, dim_features=cfg.dim_features, example=(20, 6, 12),
                            max_classes=3, task="classification")
    """)
    r = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr[-800:]


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_every_checkpoint_tensor_is_a_graph_initializer(tmp_path, task):
    """The engine injects EVERY tensor of the weights file by name, and ONNX Runtime rejects one the graph
    does not have ("Failed to find existing initializer with name m.feature_decoder.1.bias"). LimiX's one
    checkpoint holds both tasks' heads plus an imputation head, none of which a single task graph reads, so
    the exporter must declare the unread ones as (unused) initializers. ORT must still run the graph."""
    import onnx

    cfg, path, wrapper = _export(tmp_path, task)
    model = build_model(cfg.model_config, seed=0)
    tensor_map = export.postprocess(path, dict(model.state_dict()))

    assert set(tensor_map["initializers"].values()) == set(model.state_dict()), (
        "the map must cover the whole checkpoint, or the engine's injection hits a name the graph lacks")
    proto = onnx.load(str(path), load_external_data=False)
    graph_names = {i.name for i in proto.graph.initializer}
    assert set(tensor_map["initializers"]) <= graph_names
    # the unread ones really are unused: only a keep_* Identity (whose output nothing uses) consumes them
    used = {x for n in proto.graph.node if not n.name.startswith("keep_") for x in n.input}
    unused = sorted(set(tensor_map["initializers"]) - used)
    keepers = {n.input[0]: n.output[0] for n in proto.graph.node if n.name.startswith("keep_")}
    assert set(keepers) == set(unused), "every unread tensor needs exactly one keep_* Identity"
    consumed_outputs = {x for n in proto.graph.node for x in n.input}
    assert not (set(keepers.values()) & consumed_outputs), "a keep_* output feeds something"
    assert any("feature_decoder" in n for n in unused), unused
    assert len(unused) >= 10, f"expected the imputation head and the other task's heads unused, got {unused}"
    # ...and ORT accepts the graph with them present, and still agrees with PyTorch
    r = export.check_parity(path, wrapper, ((24, 7, 10),), max_classes=cfg.max_classes, task=task)
    assert r["ok"], r


# --- the classification graph must not contain `Unique` -------------------------------------------
#
# Upstream's MulticlassTargetEncoder ranks each label among the DISTINCT training labels with torch.unique,
# which exports as an ONNX `Unique`: a data-dependent output shape, absent from the MLX interpreter's op table
# (so the model would be refused on MLX by name) and hostile to graph-compiling backends.

def test_the_classification_graph_has_no_unique_op(tmp_path):
    import onnx

    _, path, _ = _export(tmp_path, "classification")
    ops = {n.op_type for n in onnx.load(str(path), load_external_data=False).graph.node}
    assert "Unique" not in ops, "the label-rank encoder still exports a data-dependent Unique"


def _labels(rng, s, t, classes, with_gaps):
    """[1, T, 1] float labels: S training labels (duplicates, optionally gaps in the class ids), then a NaN
    pad for the test rows, as the model's own y encoder produces it."""
    pool = rng.choice(np.arange(0, 3 * classes, 3 if with_gaps else 1), size=classes, replace=False)
    y = rng.choice(pool, size=s).astype(np.float32)
    x = np.full((1, t, 1), np.nan, dtype=np.float32)
    x[0, :s, 0] = y
    return torch.from_numpy(x)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("with_gaps", [False, True])
def test_the_unique_free_label_rank_equals_upstreams(seed, with_gaps):
    """The replacement must give upstream's exact ranks: duplicated labels, gaps between class ids, a class
    that appears once, and the NaN pad of the test rows (which upstream ranks as 0 because NaN > x is False)."""
    from export_limix.limix_patches import apply
    apply()
    from model import encoders as emod

    rng = np.random.default_rng(seed)
    for s, t, classes in ((12, 20, 3), (7, 9, 5), (30, 41, 10), (5, 8, 1)):
        x = _labels(rng, s, t, classes, with_gaps)
        enc = emod.MulticlassTargetEncoder()
        want = emod.MulticlassTargetEncoder._upstream_forward(enc, {"data": x.clone(), "eval_pos": s})["data"]
        got = enc({"data": x.clone(), "eval_pos": s})["data"]
        assert torch.equal(want, got), (s, t, classes, want.flatten().tolist(), got.flatten().tolist())


# --- ReLU: the MLX interpreter's op table has no `Relu` -------------------------------------------

@pytest.mark.parametrize("task", ["classification", "regression"])
def test_the_graph_has_no_relu_op(tmp_path, task):
    """`Relu` is absent from the MLX interpreter's op table, so the model would be refused on MLX by name over a
    one-node activation. clamp(x, min=0) is the same function (NaN included) and exports as `Clip`."""
    import onnx

    _, path, _ = _export(tmp_path, task)
    ops = {n.op_type for n in onnx.load(str(path), load_external_data=False).graph.node}
    assert "Relu" not in ops


def test_the_clamp_activation_equals_relu_including_nan_and_negative_zero():
    cfg = configs.fixture()
    model = build_model(cfg.model_config, seed=0)
    relus = [m for m in model.modules() if type(m).__name__ == "ReLU"]
    assert relus, "the fixture config has no ReLU to patch: the test would pass vacuously"
    x = torch.tensor([-3.0, -0.0, 0.0, 0.5, 7.0, float("nan"), float("-inf"), float("inf")])
    want = torch.relu(x)
    for m in relus:
        got = m(x)
        assert torch.equal(torch.nan_to_num(got, nan=-123.0), torch.nan_to_num(want, nan=-123.0)), (got, want)
