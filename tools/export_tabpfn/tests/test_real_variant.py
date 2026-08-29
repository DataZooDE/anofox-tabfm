"""Verify the RealTabPFN-2.5 resource-sharing claim against the ACTUAL checkpoints.

`tabpfn-v2-5-real` reuses `tabpfn-v2-5`'s bundled graph, tensor map and CUDA ext
graph, and `ExpectedWeightsHeaderShaFor` returns the same sha for both ids. The
C++ test pins those constants, but a constant matching a constant proves nothing
about the checkpoints: if Prior Labs republished `_real` with a different tensor
layout, the C++ test would still pass while the ext graph's baked offsets indexed
the wrong tensors.

This is the test that actually checks it. It needs the real weights, so it skips
unless they are cached — the same opt-in shape as
`test/sql/tabfm_real_models.test`.

    cd tools/export_tabpfn
    uv run python convert_weights.py classification --arch=v2.5
    uv run python convert_weights.py classification --arch=v2.5 --variant=real
    uv run pytest tests/test_real_variant.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
CACHE = pathlib.Path.home() / ".cache/anofox-tabfm/Prior-Labs__tabpfn_2_5@main"

# (task, expected header sha) — the values baked into
# ExpectedWeightsHeaderShaFor() in src/include/tabfm_model_spec.hpp.
CASES = [
    ("classification", "b230477af81d4ac5bff856b2f9dcc281d5b9a04d659a5dee335553f0f49897ea"),
    ("regression", "8865ee281d0172e31e1a03d1d43057ac8e69b88a27b2ce0a93ee77b865f45737"),
]


def _header_sha(path: pathlib.Path) -> str:
    """sha256 of a safetensors file's JSON header (names, shapes, dtypes)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return hashlib.sha256(f.read(n)).hexdigest()


@pytest.mark.parametrize("task,expected_sha", CASES)
def test_real_and_default_share_a_header_sha(task, expected_sha):
    default = CACHE / task / "model.safetensors"
    real = CACHE / f"{task}-real" / "model.safetensors"
    if not (default.exists() and real.exists()):
        pytest.skip(f"converted {task} weights not cached (see module docstring)")

    default_sha, real_sha = _header_sha(default), _header_sha(real)
    # The whole reason tabpfn-v2-5-real embeds no new bytes.
    assert default_sha == real_sha, (
        "the _real checkpoint no longer has the same tensor layout as _default; "
        "it can no longer share 2.5's graph, tensor map or ext graph")
    # And the shared value is the one the C++ has baked in.
    assert real_sha == expected_sha, (
        f"header sha for {task} drifted from ExpectedWeightsHeaderShaFor(): "
        f"{real_sha} != {expected_sha}")


@pytest.mark.parametrize("task,_sha", CASES)
def test_real_weights_are_genuinely_different(task, _sha):
    """Same layout, different VALUES — otherwise the entry is a pointless alias."""
    default = CACHE / task / "model.safetensors"
    real = CACHE / f"{task}-real" / "model.safetensors"
    if not (default.exists() and real.exists()):
        pytest.skip(f"converted {task} weights not cached")
    assert (hashlib.sha256(default.read_bytes()).hexdigest()
            != hashlib.sha256(real.read_bytes()).hexdigest())


@pytest.mark.parametrize("task,_sha", CASES)
def test_committed_tensor_map_covers_the_real_checkpoint(task, _sha):
    """Every key 2.5's committed map names must exist in the _real safetensors."""
    real = CACHE / f"{task}-real" / "model.safetensors"
    if not real.exists():
        pytest.skip(f"converted {task}-real weights not cached")
    tmap = json.loads(
        (REPO / f"resources/tensor_map_tabpfn25_{task}.json").read_text())
    want = set(tmap["initializers"].values())
    with open(real, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    have = set(header) - {"__metadata__"}
    assert not (want - have), sorted(want - have)[:5]
