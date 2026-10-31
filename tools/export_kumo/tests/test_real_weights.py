"""Parity with upstream on the REAL published weights (opt-in: needs the checkpoints in the engine cache).

    TABFM_REAL_WEIGHTS=1 uv run pytest tests/test_real_weights.py -s

The reference is upstream's own public API, `sdm.models.KumoTabular(...)(x_context, y_context, x_query)`,
with `num_estimators=1` and the default recipe made deterministic: the per-estimator random members
(FlipSign, ShuffleColumns, ShuffleCategories) are removed and the round-robin Choice is pinned to its
first member, Identity. Everything else is upstream's: DropConstantColumns, Standardize(eps=1e-6),
Clip(-100, 100), ClipSigma(4, "log") in float64, the target Standardize, softmax / quantile handling.

The candidate is the shipped graph run by ONNX Runtime on `engine_ref.engine_inputs(...)`, the float64
replica of the engine's preprocessing that the SQL fixture test checks against the engine. So the chain
upstream == replica+graph (here) and replica == engine (test/sql/tabfm_kumo_tabular.test) is closed.
"""

from __future__ import annotations

import os
import pathlib
import shutil

import numpy as np
import pytest
import torch

from export_kumo import engine_ref
from export_kumo.convert_weights import CACHE

pytestmark = pytest.mark.skipif(not os.environ.get("TABFM_REAL_WEIGHTS"),
                                reason="set TABFM_REAL_WEIGHTS=1 (needs the downloaded checkpoints)")

RESOURCES = pathlib.Path(__file__).resolve().parents[3] / "resources"
SIZES = [s for s in os.environ.get("KUMO_SIZES", "small,medium,large").split(",") if s]


def deterministic_recipe():
    import sdm.processing as sp

    def numerical():
        return sp.Sequential(
            sp.Cast(torch.float64), sp.DropConstantColumns(), sp.Standardize(eps=1e-6),
            sp.Clip(-100.0, 100.0), sp.ClipSigma(threshold=4.0, method="log"))

    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                numerical=[numerical()],
                categorical=[sp.AlignCategories(sort_by="value"), sp.AddCategoryCounts(min_cardinality=50),
                             sp.ToNumerical(), numerical()]),
            sp.SelectColumns(500, method="first"),
            sp.Cast(torch.float32),
        ],
        target=[sp.StypeDispatch(categorical=[sp.AlignCategories(sort_by="value")],
                                 numerical=[sp.Standardize()])],
        output=sp.TaskDispatch(
            classification=[sp.AverageEstimators(), sp.Softmax()],
            regression=[sp.SortQuantiles(), sp.AverageEstimators(trim_fraction=0.2)]),
    )


def upstream(task, size, x, y, s):
    import pandas as pd
    import sdm
    from sdm.models import KumoTabular

    model = KumoTabular(task=task, size=size, pretrained=False, device="cpu")
    sd = torch.load(CACHE / size / task / "model.ckpt", map_location="cpu", weights_only=True)
    model.models[task].load_state_dict(sd)
    model.eval()
    df = pd.DataFrame(x, columns=[f"f{j}" for j in range(x.shape[1])])
    df["target"] = y
    # Every feature numerical, as the engine sees a DOUBLE column. Left to infer_stypes, a two-valued
    # column (diabetes' `sex`) becomes categorical, which changes the input, not the model.
    overrides = {c: "numerical" for c in df.columns if c != "target"}
    overrides["target"] = "categorical" if task == "classification" else "numerical"
    stypes = sdm.infer_stypes(df, overrides=overrides)
    table = sdm.TableTensor.from_pandas(df=df, stypes=stypes, device="cpu")
    with torch.inference_mode():
        out = model(x_context=table[:s].drop_columns("target"), y_context=table[:s, "target"],
                    x_query=table[s:].drop_columns("target"), recipe=deterministic_recipe(),
                    num_estimators=1)
    return out.numerical.float().numpy()          # [Q, classes] probs, or [Q, 999] quantiles


def ours(task, size, x, y, s, tmp_path):
    import onnxruntime as ort

    st = CACHE / size / task / "model.safetensors"
    if not st.exists():
        pytest.skip(f"{st} missing: run convert_kumo_weights --size {size}")
    # ORT refuses external data outside the graph's directory, so the graph goes BESIDE the weights,
    # exactly where the engine stages it (under a name the engine never uses).
    graph = st.parent / f"graph_ext_kumo_{size}_{task}.pytest.onnx"
    shutil.copy(RESOURCES / f"graph_ext_kumo_{size}_{task}.onnx", graph)
    feed, (mean, std) = engine_ref.engine_inputs(x, y[:s], regression=task == "regression")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    try:
        (out,) = ort.InferenceSession(str(graph), so, providers=["CPUExecutionProvider"]).run(["logits"], feed)
    finally:
        graph.unlink(missing_ok=True)
    return out[0], mean, std


def datasets(task):
    from sklearn import datasets as d
    if task == "classification":
        for name, loader in [("iris", d.load_iris), ("wine", d.load_wine), ("breast_cancer", d.load_breast_cancer)]:
            ds = loader()
            yield name, ds.data, ds.target
    else:
        ds = d.load_diabetes()
        yield "diabetes", ds.data, ds.target.astype(np.float64)
        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 6))
        yield "synthetic", x, x @ np.array([1.0, -2.0, 0.5, 0.0, 3.0, 1.5]) + np.sin(x[:, 0]) + 0.1 * rng.normal(size=300)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_matches_upstream_single_estimator(task, size, tmp_path):
    torch.set_num_threads(8)
    for name, x, y in datasets(task):
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(x))
        x, y = x[perm], y[perm]
        s = int(0.7 * len(x))
        ref = upstream(task, size, x, y, s)
        out, mean, std = ours(task, size, x, y, s, tmp_path / name)
        if task == "classification":
            k = ref.shape[1]
            logits = out[s:, :k]
            prob = np.exp(logits - logits.max(-1, keepdims=True))
            prob /= prob.sum(-1, keepdims=True)
            agree = (prob.argmax(-1) == ref.argmax(-1)).mean()
            acc = (prob.argmax(-1) == y[s:]).mean()
            diff = np.abs(prob - ref).max()
            print(f"\n{size} {name}: labels agree {agree:.4f}, max |prob diff| {diff:.2e}, query acc {acc:.4f}")
            assert agree == 1.0
            assert diff < 2e-3
        else:
            point = out[s:, 0] * std + mean
            ref_point = ref.mean(-1)
            diff = np.abs(point - ref_point).max()
            spread = y.max() - y.min()
            r2 = 1 - ((point - y[s:]) ** 2).sum() / ((y[s:] - y[s:].mean()) ** 2).sum()
            print(f"\n{size} {name}: max |diff| {diff:.3e} on a target spread of {spread:.1f}, "
                  f"corr {np.corrcoef(point, ref_point)[0, 1]:.6f}, query R2 {r2:.4f}")
            assert diff < 1e-3 * spread
