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
        unused = sorted(keys - set(tensor_map["initializers"].values()))
    # buffers the graph constant-folded away are legitimate; list them so a real gap cannot hide in the count
    print(task, "- checkpoint tensors the graph does not read:", unused)
    # Exactly these are unread: the imputation head (both tasks) and the OTHER task's target encoder/decoder.
    # Anything else unread would be a checkpoint tensor the graph silently left at its trace-time value.
    other_task = "reg_y_" if task == "classification" else "cls_y_"
    assert all(k.startswith(("feature_decoder.", other_task)) for k in unused), unused
    assert any(k.startswith("feature_decoder.") for k in unused)
    assert any(k.startswith(other_task) for k in unused)
    assert len(keys) == 137, f"the checkpoint has {len(keys)} tensors, the spike notes say 137"
