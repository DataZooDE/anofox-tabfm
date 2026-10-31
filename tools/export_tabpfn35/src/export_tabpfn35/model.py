"""Build (random-init) or load (released weights) a TabPFN-3.5 model."""

from __future__ import annotations

import pathlib

import torch

import tabpfn.architectures.tabpfn_v3_5 as v35


def wake(model, seed: int = 0, scale: float = 0.05):
    """Give every all-zero parameter small random values.

    Upstream zero-initialises every attention out-projection and MLP output layer,
    so a freshly built model is the identity through every block: a parity test on
    it would pass whatever the attention did. Waking them makes the fixture and the
    parity gate sensitive to the attention paths (tools/export_tabpfn's
    mask_parity.py does the same for v3).
    """
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            if p.numel() and bool((p == 0).all()):
                p.copy_(torch.randn(p.shape, generator=g) * scale)
    return model


def build_random_model(model_kwargs: dict, seed: int = 0):
    """Random-init TabPFN-3.5 at the given dims. No checkpoint bytes anywhere."""
    torch.manual_seed(seed)
    cfg = v35.TabPFNV3p5Config(**model_kwargs)
    model = v35.TabPFNV3p5(config=cfg)
    return wake(model, seed).eval()


def load_real_model(path: str | pathlib.Path, task: str = "classification"):
    """Load a released checkpoint through upstream's own loader (parity only)."""
    from tabpfn.model_loading import load_model

    est = "classifier" if task == "classification" else "regressor"
    model, _criterion, _cfg, _inf = load_model(path=pathlib.Path(path), estimator_type=est)
    return model.eval()
