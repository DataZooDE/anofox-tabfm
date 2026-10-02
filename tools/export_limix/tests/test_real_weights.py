"""LimiX-2M on the REAL checkpoint: the contracts random weights cannot check.

Skipped unless TABFM_REAL_WEIGHTS is set. It downloads stable-ai/LimiX-2M (about 9.5 MB) into the Hugging
Face cache, never into this repo. The weights are under the Stable AI Technology Co., Ltd. License v1.0
(Apache-2.0 + a Section 10 attribution clause); this is internal evaluation, which that clause exempts. Eager PyTorch only: the exported graph is covered by test_export.py, and the
real-weights ORT comparison lives in the spike notes.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("TABFM_REAL_WEIGHTS"),
                                reason="needs the real LimiX-2M checkpoint (set TABFM_REAL_WEIGHTS=1)")

from sklearn.datasets import load_diabetes, make_classification  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from export_limix import configs  # noqa: E402
from export_limix.limix_patches import ExportWrapper, build_model  # noqa: E402


@pytest.fixture(scope="module")
def wrappers():
    from huggingface_hub import hf_hub_download
    sd = torch.load(hf_hub_download("stable-ai/LimiX-2M", "LimiX-2M.ckpt"), map_location="cpu",
                    weights_only=False)["state_dict"]
    out = {}
    for task in ("classification", "regression"):
        m = build_model(configs.real().model_config, seed=0)
        m.load_state_dict(sd, strict=True)
        out[task] = ExportWrapper(m.eval(), task=task).eval()
    return out


def test_context_rows_are_in_context_values_not_a_constant(wrappers):
    """5-class problem: decoding every row from one pass scored context accuracy 0.111 here, and the
    second-pass route 1.000 (spike measurement). Pin the number that chose the route."""
    X, y = make_classification(n_samples=400, n_features=20, n_informative=8, n_classes=5, random_state=1)
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    x = torch.from_numpy(np.vstack([Xtr, Xte]).astype(np.float32))[None]
    classes = np.unique(ytr)
    labels = np.searchsorted(classes, ytr)
    with torch.no_grad():
        out = wrappers["classification"](x, torch.from_numpy(labels.astype(np.float32))[None])
    fitted = out[0, :len(ytr), :len(classes)].argmax(-1).numpy()
    assert (fitted == labels).mean() > 0.95
    assert len(set(fitted.tolist())) == len(classes)


def test_regression_takes_raw_targets(wrappers):
    """Diabetes has a target scale of 79 (mean ~150). Called un-normalised the model scored query R2
    -4.2; with the standardisation in the wrapper it scores ~0.36, as upstream does on the same split."""
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    x = torch.from_numpy(np.vstack([Xtr, Xte]).astype(np.float32))[None]
    with torch.no_grad():
        out = wrappers["regression"](x, torch.from_numpy(ytr.astype(np.float32))[None])[0, :, 0].numpy()
    r2 = 1 - ((out[len(ytr):] - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    assert r2 > 0.2, f"query R2 {r2:.3f}: the target is not being standardised"


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_converted_weights_map_onto_the_exported_graph_strictly(tmp_path, task):
    """The converted safetensors must cover EVERY initializer the exported real-config graph takes from
    the checkpoint: same key, same shape (transposed where the map says so), float32, and nothing the graph
    wants is missing. A silent gap here would be an initializer left at its random trace-time value."""
    import onnx
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from export_limix import convert, export
    from export_limix.limix_patches import ExportWrapper

    ckpt = hf_hub_download("stable-ai/LimiX-2M", "LimiX-2M.ckpt")
    weights = tmp_path / "model.safetensors"
    convert.convert(ckpt, weights)

    cfg = configs.real()
    model = build_model(cfg.model_config, seed=0)
    graph = tmp_path / f"graph_limix_{task}.onnx"
    export.export_graph(model, graph, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                        dim_features=cfg.dim_features, example=cfg.example, max_classes=cfg.max_classes,
                        task=task)
    tensor_map = export.postprocess(graph, dict(model.state_dict()))

    proto = onnx.load(str(graph), load_external_data=False)
    shapes = {i.name: tuple(i.dims) for i in proto.graph.initializer}
    with safe_open(str(weights), framework="np") as f:
        keys = set(f.keys())
        assert tensor_map["initializers"], "the map is empty"
        for init, key in tensor_map["initializers"].items():
            assert key in keys, f"{init}: key {key!r} is not in the converted safetensors"
            shape = tuple(f.get_slice(key).get_shape())
            want = shapes[init]
            if tensor_map["transforms"].get(init) == "transpose":
                shape = shape[::-1]
            assert shape == want, f"{init}: safetensors {shape} != graph {want}"
            assert f.get_tensor(key).dtype == np.float32, f"{key} is not float32"
        # every checkpoint tensor is now a graph initializer (the engine injects them all by name); which of
        # them the graph actually READS is what distinguishes a mapped weight from a declared-but-unused one
        assert set(tensor_map["initializers"].values()) == keys, "the map must cover the whole checkpoint"
    consumed = {x for n in proto.graph.node if not n.name.startswith("keep_") for x in n.input}
    unused = sorted(k for i, k in tensor_map["initializers"].items() if i not in consumed)
    print(task, "- checkpoint tensors the graph declares but does not read:", unused)
    # Exactly these are unread: the imputation head (both tasks) and the OTHER task's target encoder/decoder.
    # Anything else unread would be a checkpoint tensor the graph silently ignores.
    other_task = "reg_y_" if task == "classification" else "cls_y_"
    assert all(k.startswith(("feature_decoder.", other_task)) for k in unused), unused
    assert any(k.startswith("feature_decoder.") for k in unused)
    assert any(k.startswith(other_task) for k in unused)
    assert len(keys) == 137, f"the checkpoint has {len(keys)} tensors, the spike notes say 137"


def test_a_feature_value_exactly_at_the_training_mean_is_predicted_badly_by_upstream_too(wrappers):
    """LimiX encodes a standardised value as (sign, decimal exponent, mantissa) and an EXACT zero as its own
    token. On a symmetric table a row sitting exactly on the training mean standardises to exactly 0.0, and
    the real model gets that row badly wrong: 25.1 against a truth of 37.3, while its neighbours at z = +-0.17
    are off by ~0.3-0.4. The engine reproduces it (it matches upstream eager on this table to float noise), so
    this is the MODEL's behaviour. It is pinned here so a change in upstream or in the exporter is noticed, and
    so docs/REAL_MODELS.md has a measured sentence to point at."""
    i = np.arange(100)
    tr, te = i[i % 5 != 4], i[i % 5 == 4]
    x = torch.tensor(np.stack([np.concatenate([tr, te]), np.concatenate([tr, te]) % 7], 1), dtype=torch.float32)[None]
    y = torch.tensor(tr * 0.7 + 3, dtype=torch.float32)[None]
    with torch.no_grad():
        out = wrappers["regression"](x, y)[0, :, 0].numpy()[len(tr):]
    err = np.abs(out - (te * 0.7 + 3))
    assert tr.mean() == 49.0 and 49 in te, "the table must put a held-out row exactly on the training mean"
    at_mean = err[list(te).index(49)]
    neighbours = np.abs(err[[list(te).index(44), list(te).index(54)]])
    assert at_mean > 5.0, f"the exact-mean row is now fine ({at_mean:.2f}); update the docs, this was a known flaw"
    assert neighbours.max() < 1.0, neighbours

    # ...and the same model on a holdout with NO row on the mean is excellent
    tr2, te2 = i[i % 5 != 3], i[i % 5 == 3]
    x2 = torch.tensor(np.stack([np.concatenate([tr2, te2]), np.concatenate([tr2, te2]) % 7], 1),
                      dtype=torch.float32)[None]
    y2 = torch.tensor(tr2 * 0.7 + 3, dtype=torch.float32)[None]
    with torch.no_grad():
        out2 = wrappers["regression"](x2, y2)[0, :, 0].numpy()[len(tr2):]
    assert np.abs(out2 - (te2 * 0.7 + 3)).max() < 1.5


def test_the_ort_outlier_is_the_tokenizers_near_zero_sensitivity_not_an_export_defect(tmp_path):
    """The spike left one unexplained number: at (T=33, H=9, S=21) the exported graph in ORT differed from
    PyTorch by 1.2e-3 where every other shape agrees to ~1e-5. Measured here, on the real weights:

      * the input that produces it (make_feed seed 5) is chaotic in EAGER PyTorch too: 1-ulp input noise moves
        its logits by ~8e-3 median (seed 0: 3e-5), so ORT and torch rounding differences are amplified, not
        introduced, by the graph;
      * the amplifier is the numeric tokenizer: it encodes |z| as (decimal exponent, mantissa), so a
        standardised value near zero is hypersensitive in proportion to 1/|z|. This input contains one at
        3.2e-6 (the next smallest of 200 inputs is 2.3e-5, and only this one exceeds 5e-4);
      * nudging ONE raw element by 0.37 (min |z| -> 2.3e-4) takes the ORT-vs-torch gap from 5.4e-3 to 1.4e-5.

    So it is a property of the model's input encoding, not of the export. (An earlier guess, the float16
    round-trip in the target embedding, was refuted: turning every fp16 cast into fp32 left the sensitivity
    unchanged, and for classification that embedding is an exact table lookup.)"""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    from export_limix import export

    sd = torch.load(hf_hub_download("stable-ai/LimiX-2M", "LimiX-2M.ckpt"), map_location="cpu",
                    weights_only=False)["state_dict"]
    cfg = configs.real()
    model = build_model(cfg.model_config, seed=0)
    model.load_state_dict(sd, strict=True)
    model.eval()
    path = tmp_path / "g.onnx"
    wrapper = export.export_graph(model, path, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                                  dim_features=cfg.dim_features, example=cfg.example,
                                  max_classes=cfg.max_classes, task="classification")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    tok = dict(model.named_modules())["encoder_x.0.numeric_mlp"]

    def delta_and_min_z(feed):
        store = {}
        hook = tok.register_forward_pre_hook(lambda m, i: store.__setitem__("x", i[0].detach().double()))
        with torch.no_grad():
            ref = wrapper(torch.from_numpy(feed["x"]), torch.from_numpy(feed["y"])).numpy()
        hook.remove()
        (got,) = sess.run(["logits"], feed)
        z = store["x"].abs().flatten()
        return float(np.abs(got - ref).max()), float(z[z > 0].min())

    t, h, s = 33, 9, 21
    ordinary, _ = delta_and_min_z(export.make_feed(t, h, s, 10, seed=0))
    assert ordinary < 2e-4, ordinary

    feed = export.make_feed(t, h, s, 10, seed=5)
    outlier, min_z = delta_and_min_z(feed)
    assert min_z < 1e-5 and outlier > 1e-3, (outlier, min_z)

    nudged = {k: v.copy() for k, v in feed.items()}
    nudged["x"][0, 0, 5] += 0.37
    fixed, min_z2 = delta_and_min_z(nudged)
    assert min_z2 > 1e-4 and fixed < 1e-4 and outlier / fixed > 50, (fixed, min_z2)
