"""EXAONE Tabular on the REAL checkpoints: the contracts random weights cannot check.

Skipped unless TABFM_REAL_WEIGHTS is set. It downloads LG-AI-Research/EXAONE-Tabular (two ~84 MB files) at
the revision the registry pins into the Hugging Face cache, never into this repo. The weights are under the
EXAONE AI Model License Agreement 1.2 - NC (research and evaluation use only); running these tests is
evaluation.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("TABFM_REAL_WEIGHTS"),
                                reason="needs the real EXAONE Tabular checkpoints (set TABFM_REAL_WEIGHTS=1)")

from safetensors.torch import load_file  # noqa: E402
from sklearn.datasets import load_diabetes, load_wine  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from export_exaone import configs, export  # noqa: E402
from export_exaone.exaone_patches import ExportWrapper, build_model, model_forward, upstream_forward  # noqa: E402

REVISION = "093ad1c2613e1e95d26df8610451d2e02ed53cbe"
FILES = {"classification": "exaone-tabular-classifier-v1_default.safetensors",
         "regression": "exaone-tabular-regressor-v1_default.safetensors"}
SHA256 = {"classification": "2aa723e024e5c56c778d6c79cc3cee7675d1b6a03c06fa08c80e620e228b224d",
          "regression": "24891e14e649c6509f07ee62b5583e06544b68eaf9a9195ee58777eb9cf6fa0d"}


@pytest.fixture(scope="module")
def models():
    import hashlib

    from huggingface_hub import hf_hub_download
    out = {}
    for task, name in FILES.items():
        path = hf_hub_download("LG-AI-Research/EXAONE-Tabular", name, revision=REVISION)
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == SHA256[task]
        cfg = configs.get("real", task)
        m = build_model(task, cfg.model_config, cfg.regression_config)
        m.load_state_dict(load_file(path), strict=True)
        out[task] = m.eval()
    return out


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_the_export_forward_is_upstreams_forward_on_the_real_weights(models, task):
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, 80, 9, generator=gen)
    y = torch.randint(0, 4, (1, 50), generator=gen).float() if task == "classification" else torch.randn(1, 50)
    with torch.no_grad():
        ours = model_forward(models[task], x, y, 50, regression=task == "regression")
    ref = upstream_forward(models[task], x, y, 50)
    assert (ours - ref).abs().max() < 1e-4


def test_the_graph_on_the_real_weights_matches_upstream(tmp_path, models):
    """ORT on the exported graphs (real weights inside) vs upstream eager: wine labels and diabetes points."""
    import onnxruntime as ort

    for task in ("classification", "regression"):
        cfg = configs.get("real", task)
        graph = tmp_path / f"{task}.onnx"
        export.export_graph(models[task], graph, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                            dim_features=cfg.dim_features, example=cfg.example, task=task)
        sess = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
        loader = load_wine if task == "classification" else load_diabetes
        X, y = loader(return_X_y=True)
        xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        x = np.vstack([xtr, xte]).astype(np.float32)[None]
        got = sess.run(["logits"], {"x": x, "y": ytr.astype(np.float32)[None]})[0][0, len(ytr):]
        with torch.no_grad():
            ref = ExportWrapper(models[task], task)(torch.from_numpy(x),
                                                    torch.from_numpy(ytr.astype(np.float32))[None])[0, len(ytr):]
        if task == "classification":
            assert (got[:, :3].argmax(-1) == ref[:, :3].argmax(-1).numpy()).all()
            assert (got[:, :3].argmax(-1) == yte).mean() > 0.9
        else:
            assert np.abs(got[:, 0] - ref[:, 0].numpy()).max() < 1e-2
            r2 = 1 - ((got[:, 0] - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
            assert r2 > 0.3, r2
