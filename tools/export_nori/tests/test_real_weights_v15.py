"""Nori 1.5 on the REAL checkpoint: the contracts random weights cannot check.

Skipped unless TABFM_REAL_WEIGHTS is set. It uses Synthefy/Nori-1-5 (83 MB, Apache-2.0, ungated) at the revision
the registry pins: from $NORI15_CKPT when that names a file whose sha256 matches, else from the Hugging Face
cache. Nothing is written into this repo.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("TABFM_REAL_WEIGHTS"),
                                reason="needs the real Nori 1.5 checkpoint (set TABFM_REAL_WEIGHTS=1)")

from sklearn.datasets import load_diabetes  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from export_nori import configs, export  # noqa: E402
from export_nori.nori_patches import load_real, unpatched  # noqa: E402

REVISION = "79afc5a65b2775d0c62977211a63620ce87846c7"
CKPT_SHA256 = "b1313b45225066f74a78e99d5cd56dbd0015bdbeadf948252cf82d2a962766a2"
CKPT_BYTES = 83_176_401
# Header of the deterministic safetensors convert_nori_weights writes from this checkpoint: what
# ExpectedWeightsHeaderShaFor("nori-1-5", "regression") pins for the bundled ext graph.
SAFETENSORS_HEADER_SHA256 = "6da735b26d65fd2281525b127938f45c48d9eeef7bc20156ba6a87caaeed6029"


@pytest.fixture(scope="module")
def ckpt():
    local = os.environ.get("NORI15_CKPT")
    if local and os.path.exists(local):
        if hashlib.sha256(open(local, "rb").read()).hexdigest() != CKPT_SHA256:
            pytest.fail(f"{local} is not Synthefy/Nori-1-5@{REVISION[:8]} nori.pt (sha256 mismatch)")
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download("Synthefy/Nori-1-5", "nori.pt", revision=REVISION)


@pytest.fixture(scope="module")
def model(ckpt):
    return load_real(ckpt)


def test_the_checkpoint_is_the_config_the_exporter_transcribed(ckpt):
    assert os.path.getsize(ckpt) == CKPT_BYTES
    obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert set(obj) == {"model_config", "model_state_dict", "inference_config"}
    cfg = dict(obj["model_config"])
    cfg["mask_prediction"] = False
    # the file lists the 999 levels explicitly; they ARE the uniform grid the exporter generates
    assert cfg["decoder_config"].pop("regression_quantiles") == \
        configs.real15().model_config["decoder_config"].pop("regression_quantiles")
    want = configs.real15().model_config
    want["decoder_config"].pop("regression_quantiles")
    assert cfg == want
    sd = obj["model_state_dict"]
    assert len(sd) == 381 and sum(t.numel() for t in sd.values()) == 20_757_833
    assert {t.dtype for t in sd.values()} == {torch.float32}


def test_the_released_model_is_upstreams_forward(model):
    """Patched forward = upstream eager forward on the real weights."""
    x = torch.randn(1, 60, 9, generator=torch.Generator().manual_seed(0))
    y = torch.randn(1, 40, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ours = model(x, y, eval_pos=40, task_type="reg")
        with unpatched():
            theirs = model(x, y, eval_pos=40, task_type="reg")
    assert torch.allclose(ours, theirs, atol=1e-5, rtol=1e-5), (ours - theirs).abs().max()


def test_the_graph_on_the_real_weights_is_upstreams_point_estimate(tmp_path, model):
    """ORT on the exported graph (with the real weights in it) vs upstream's own single forward and quantile
    mean, on diabetes (a real regression table, odd feature width 10, 309 context rows)."""
    import onnxruntime as ort
    from synthefy_nori.model.quantile_dist import quantile_dist_mean_simple

    cfg = configs.real15()
    graph = tmp_path / "g.onnx"
    w = export.export_graph(model, graph, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                            dim_features=cfg.dim_features, example=cfg.example)
    sess = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    X, yv = load_diabetes(return_X_y=True)
    xtr, xte, ytr, yte = train_test_split(X, yv, test_size=0.3, random_state=0)
    x = np.vstack([xtr, xte]).astype(np.float32)[None]
    got = sess.run(["logits"], {"x": x, "y": ytr.astype(np.float32)[None]})[0][0, len(ytr):, 0]
    mu, sd = ytr.mean(), ytr.std()
    with torch.no_grad(), unpatched():
        q = model(torch.from_numpy(x), torch.from_numpy(((ytr - mu) / sd).astype(np.float32))[None],
                  eval_pos=len(ytr), task_type="reg")
        ref = quantile_dist_mean_simple(q[0].double(), torch.tensor(model.regression_quantiles,
                                                                    dtype=torch.float64)).numpy() * sd + mu
    gap = float(np.abs(got - ref).max())
    r2 = 1 - ((got - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    r2_ref = 1 - ((ref - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    print(f"diabetes: ORT vs upstream max |delta| {gap:.3g} (target std {yv.std():.1f}); R2 {r2:.4f} vs {r2_ref:.4f}")
    assert gap < 0.05 * yv.std(), gap
    assert abs(r2 - r2_ref) < 5e-3 and r2 > 0.3, (r2, r2_ref)


def test_the_converter_is_deterministic_and_matches_the_pinned_header(tmp_path, ckpt):
    import struct

    from export_nori.convert import convert

    shas = []
    for name in ("a.safetensors", "b.safetensors"):
        assert convert(ckpt, tmp_path / name) == 381
        with open(tmp_path / name, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            shas.append(hashlib.sha256(f.read(n)).hexdigest())
    assert (tmp_path / "a.safetensors").read_bytes() == (tmp_path / "b.safetensors").read_bytes()
    assert shas[0] == SAFETENSORS_HEADER_SHA256


