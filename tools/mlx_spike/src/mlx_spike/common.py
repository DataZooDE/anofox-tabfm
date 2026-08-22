"""Shared plumbing for the MLX spikes: paths, weights, inputs, comparison.

Everything here is deliberately explicit about *where a number came from*.
The GPU-hardening work's central lesson was that a comparison which silently
runs both sides on the same backend passes while testing nothing, so every
loader below reports what it actually opened and every runner reports what
actually executed it.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
RESOURCES = REPO_ROOT / "resources"
CACHE_ROOT = Path(os.environ.get("ANOFOX_TABFM_CACHE", Path.home() / ".cache" / "anofox-tabfm"))

# The registry's cache slug for mitra (repo '/'->'__', then '@' + revision).
MITRA_CACHE = {
    "classification": CACHE_ROOT / "autogluon__mitra-classifier@main",
    "regression": CACHE_ROOT / "autogluon__mitra-regressor@main",
}

# ExpectedWeightsHeaderShaFor("mitra", task) in src/include/tabfm_model_spec.hpp.
# The ext graph bakes absolute byte offsets into the safetensors data section;
# a matching header sha is what proves those offsets index THIS download.
MITRA_HEADER_SHA = {
    "classification": "cb1a261bb3d9ca505e0db66e21df85bec6777f7e104247d4a71a8eb8d8b3b96a",
    "regression": "44f9293fa2d81ccf56dffab09600ec4f775c5c30bbc2af551cf4f6b2cf889f01",
}

# tools/export_mitra/src/export_mitra/configs.py, config="real".
MITRA_DIMS = {"dim": 512, "n_layers": 12, "n_heads": 4}
MITRA_DIM_OUTPUT = {"classification": 10, "regression": 1}


class SpikeError(RuntimeError):
    """A spike precondition failed. Always names the fixing command."""


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
@dataclass
class Safetensors:
    """A memory-mapped safetensors file, sliced by the on-disk header."""

    path: Path
    header: dict
    data_offset: int
    header_sha: str
    _mm: np.memmap

    @property
    def names(self) -> list[str]:
        return sorted(k for k in self.header if k != "__metadata__")

    def get(self, name: str) -> np.ndarray:
        entry = self.header[name]
        if entry["dtype"] != "F32":
            raise SpikeError(f"{name}: expected F32, got {entry['dtype']}")
        begin, end = entry["data_offsets"]
        raw = self._mm[self.data_offset + begin : self.data_offset + end]
        return raw.view(np.float32).reshape(entry["shape"])


def open_safetensors(path: Path) -> Safetensors:
    if not path.exists():
        raise SpikeError(
            f"weights not found at {path}. Fetch them with:\n"
            f"  curl -L --create-dirs -o {path} \\\n"
            f"    https://huggingface.co/autogluon/mitra-classifier/resolve/main/model.safetensors"
        )
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_len)
    header = json.loads(header_bytes)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    return Safetensors(
        path=path,
        header=header,
        data_offset=8 + header_len,
        header_sha=hashlib.sha256(header_bytes).hexdigest(),
        _mm=mm,
    )


def load_mitra_weights(task: str = "classification") -> Safetensors:
    st = open_safetensors(MITRA_CACHE[task] / "model.safetensors")
    expected = MITRA_HEADER_SHA[task]
    if st.header_sha != expected:
        raise SpikeError(
            f"safetensors header sha {st.header_sha} != {expected} baked into "
            f"tabfm_model_spec.hpp. The ext graph's offsets do not index this "
            f"file; re-download or regenerate the graph."
        )
    return st


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass
class Workload:
    """One forward-pass input, matching the engine's tensor contract.

    x is [1, t, h], y is [1, t]; rows [0, train_size) are support (labelled),
    rows [train_size, t) are query. d is the count of *real* feature columns
    (h may be padded); cat_mask is unused by mitra but part of the ABI.
    """

    x: np.ndarray
    y: np.ndarray
    train_size: int
    d: int

    @property
    def t(self) -> int:
        return self.x.shape[1]

    @property
    def h(self) -> int:
        return self.x.shape[2]


def make_workload(t: int, h: int, train_size: int, d: int | None = None, *, n_classes: int = 3,
                  seed: int = 0) -> Workload:
    """A deterministic synthetic table. Not random noise: the support rows carry
    a learnable signal so a correct forward produces confident, *differentiated*
    logits -- comparing two backends on inputs whose true answer is uniform
    would agree even if both were broken."""
    rng = np.random.default_rng(seed)
    d = h if d is None else d
    x = rng.standard_normal((1, t, h)).astype(np.float32)
    # A linear rule over the first two real columns decides the class.
    score = x[0, :, 0] + 0.5 * x[0, :, min(1, d - 1)]
    edges = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    labels = np.digitize(score, edges).astype(np.float32)
    y = labels[None, :].copy()
    # Query labels are not read by the model, but the engine passes zeros.
    y[0, train_size:] = 0.0
    if h > d:  # padded feature columns are zero, as the engine pads them
        x[0, :, d:] = 0.0
    return Workload(x=x, y=y, train_size=train_size, d=d)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class Comparison:
    max_abs: float
    max_rel: float
    max_rel_strict: float
    prob_max_abs: float
    argmax_agreement: float
    n: int

    def describe(self) -> str:
        return (f"logit_max_abs={self.max_abs:.3e} prob_max_abs={self.prob_max_abs:.3e} "
                f"argmax_agreement={self.argmax_agreement:.4f} over n={self.n}")


def compare(reference: np.ndarray, candidate: np.ndarray) -> Comparison:
    """Compare two logit tensors. `reference` is CPU by definition.

    Three metrics, because the obvious one is misleading here:

    - `max_rel_strict` is the project's existing metric (equivalence.py):
      max per-element |a-b|/max(|a|,1e-12). On logits it is close to useless --
      logits cross zero, so a negligible absolute difference next to a
      near-zero logit produces an unbounded relative error. Measured: the torch
      reference implementation scores 9.3e-04 against its OWN ONNX export under
      this metric, i.e. the bar of 1e-4 is one the definition cannot clear.
      Kept so the claim stays checkable rather than asserted.
    - `prob_max_abs` is the same difference after softmax -- the quantity the
      extension actually hands back to SQL. This is the metric with product
      meaning, and it is bounded in [0, 1].
    - `argmax_agreement` is the predicted class. A backend that disagrees here
      changes answers, which is the property the whole exercise exists to hold.
    """
    if reference.shape != candidate.shape:
        raise SpikeError(f"shape mismatch: reference {reference.shape} vs candidate {candidate.shape}")
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    diff = np.abs(a - b)
    # Floored so near-zero logits do not manufacture an enormous relative error
    # out of a negligible absolute one; max_rel_strict keeps the unfloored view.
    denom = np.maximum(np.abs(a), 1e-3)
    if a.ndim >= 2 and a.shape[-1] > 1:
        agree = float(np.mean(np.argmax(a, axis=-1) == np.argmax(b, axis=-1)))
        prob = float(np.abs(softmax(a) - softmax(b)).max())
    else:
        agree = 1.0        # regression: no argmax to agree on...
        prob = float(diff.max())  # ...and no softmax; fall back to the raw value
    return Comparison(
        max_abs=float(diff.max()),
        max_rel=float((diff / denom).max()),
        max_rel_strict=float((diff / np.maximum(np.abs(a), 1e-12)).max()),
        prob_max_abs=prob,
        argmax_agreement=agree,
        n=int(a.size),
    )
