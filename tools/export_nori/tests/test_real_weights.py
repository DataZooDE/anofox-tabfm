"""Nori-30M on the REAL checkpoint: the contracts random weights cannot check.

Skipped unless TABFM_REAL_WEIGHTS is set. It downloads Synthefy/Nori-30M (117 MB, Apache-2.0, ungated) at
the revision the registry pins into the Hugging Face cache, never into this repo.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("TABFM_REAL_WEIGHTS"),
                                reason="needs the real Nori-30M checkpoint (set TABFM_REAL_WEIGHTS=1)")

from sklearn.datasets import load_diabetes  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from export_nori import configs, export  # noqa: E402
from export_nori.nori_patches import ExportWrapper, load_real, unpatched  # noqa: E402

REVISION = "63c9f7facf9fb32c37ce3fc2fba331d524696318"


@pytest.fixture(scope="module")
def ckpt():
    from huggingface_hub import hf_hub_download
    return hf_hub_download("Synthefy/Nori-30M", "nori.pt", revision=REVISION)


@pytest.fixture(scope="module")
def model(ckpt):
    return load_real(ckpt)


def test_the_checkpoint_is_the_config_the_exporter_transcribed(ckpt):
    obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert set(obj) == {"model_config", "ema_state_dict"}
    cfg = dict(obj["model_config"])
    cfg["mask_prediction"] = False
    assert cfg == configs.real().model_config
    sd = obj["ema_state_dict"]
    assert len(sd) == 773 and sum(t.numel() for t in sd.values()) == 29_206_833


def test_the_released_model_is_upstreams_forward(model):
    """Patched forward = upstream eager forward on the real weights, bit for bit."""
    x = torch.randn(1, 60, 9, generator=torch.Generator().manual_seed(0))
    y = torch.randn(1, 40, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ours = model(x, y, eval_pos=40, task_type="reg")
        with unpatched():
            theirs = model(x, y, eval_pos=40, task_type="reg")
    assert torch.equal(ours, theirs)


def test_the_graph_on_the_real_weights_is_upstreams_point_estimate(tmp_path, model):
    """ORT on the exported graph (with the real weights in it) vs upstream's own single forward and
    quantile mean, on diabetes. The gap is float32 noise amplified by the decimal tokenizer: 1e-7 relative
    input noise moves upstream's own eager output by a comparable amount, which this test also pins."""
    import onnxruntime as ort
    from synthefy_nori.model.quantile_dist import quantile_dist_mean_simple

    cfg = configs.real()
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
        noisy = w(torch.from_numpy(x * (1 + 1e-7 * np.random.default_rng(0).standard_normal(x.shape))
                                   .astype(np.float32)), torch.from_numpy(ytr.astype(np.float32))[None])
    gap = np.abs(got - ref).max()
    self_noise = np.abs(noisy[0, len(ytr):, 0].numpy() - w(torch.from_numpy(x), torch.from_numpy(
        ytr.astype(np.float32))[None])[0, len(ytr):, 0].detach().numpy()).max()
    print(f"diabetes: ORT vs upstream {gap:.3g}, upstream's own 1e-7-noise response {self_noise:.3g}")
    assert gap < 0.25, gap                       # target std ~71
    assert gap < 5 * self_noise + 1e-3, (gap, self_noise)
    r2 = 1 - ((got - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    r2_ref = 1 - ((ref - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    assert abs(r2 - r2_ref) < 1e-3 and r2 > 0.3, (r2, r2_ref)


def test_the_point_estimate_is_the_mean_not_the_median(model):
    """The documented choice: the in-graph point estimate is upstream's default ("mean"); on a skewed
    target the mean and the median (upstream's other option) differ, so this pins which one ships."""
    rng = np.random.default_rng(3)
    xtr = rng.standard_normal((200, 3))
    ytr = np.exp(xtr[:, 0] + 0.5 * rng.standard_normal(200))          # right-skewed
    xte = rng.standard_normal((50, 3))
    x = torch.from_numpy(np.vstack([xtr, xte]).astype(np.float32))[None]
    with torch.no_grad():
        ours = ExportWrapper(model)(x, torch.from_numpy(ytr.astype(np.float32))[None])[0, 200:, 0].numpy()
        mu, sd = ytr.mean(), ytr.std()
        q = model(x, torch.from_numpy(((ytr - mu) / sd).astype(np.float32))[None], eval_pos=200,
                  task_type="reg")[0].double()
    mean = (q.sort(-1).values.mean(-1).numpy()) * sd + mu
    median = q.sort(-1).values[:, q.shape[-1] // 2].numpy() * sd + mu
    assert np.abs(ours - mean).max() < 0.05 * np.abs(mean).max()
    assert np.abs(ours - median).mean() > np.abs(ours - mean).mean()


def test_the_masked_rocm_contract_on_the_real_weights(model):
    """Padded + masked == positional on the real weights, and every negative control is caught."""
    from export_nori.mask_parity import gate

    res = gate(model)
    assert res["ok"], res


def test_the_converter_is_deterministic_and_matches_the_rocm_graph(tmp_path, ckpt):
    """convert_nori_weights' header is the one ExpectedWeightsHeaderShaFor pins for the bundled ROCm graph."""
    import hashlib
    import struct

    from export_nori.convert import convert

    shas = []
    for name in ("a.safetensors", "b.safetensors"):
        convert(ckpt, tmp_path / name)
        with open(tmp_path / name, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            shas.append(hashlib.sha256(f.read(n)).hexdigest())
    assert (tmp_path / "a.safetensors").read_bytes() == (tmp_path / "b.safetensors").read_bytes()
    assert shas[0] == "49797138da9ce6d7bb76895cb00514bc6967732051811c217548f32c75a02363"
