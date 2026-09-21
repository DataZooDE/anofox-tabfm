"""MODL-04: pytest for the tabpfn_v2 distribution contract validator.

Tests:
  1. Positive: check_tabpfn_v2_contract passes on the committed fixture.
  2. Negative: mutated (non-increasing) borders array fails the check.
  3. Negative: missing 'borders' output in a minimal fake graph fails the check.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from parity.check_tabpfn_v2 import check_tabpfn_v2_contract

# Repo root: tests/ is 3 levels up from tools/parity/tests/
_REPO_ROOT = pathlib.Path(__file__).parents[3]
_FIXTURE_DIR = _REPO_ROOT / "test" / "fixtures" / "tabpfn_v2"

K_FIXTURE = 16
H_FIXTURE = 8


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_minimal_graph(
    K: int,
    H: int,
    borders: np.ndarray,
    include_borders_output: bool = True,
) -> bytes:
    """Build a minimal valid ONNX graph for testing.

    Graph: x [1,T,H] → Reshape [T,H] → MatMul [T,K] → logits [T,K]
    Plus optional Constant node wired to borders output.
    """
    W = np.zeros((H, K), dtype=np.float32)
    reshape_shape = np.array([-1, H], dtype=np.int64)

    x_in       = helper.make_tensor_value_info("x",          TensorProto.FLOAT, [1, None, H])
    y_in       = helper.make_tensor_value_info("y",          TensorProto.FLOAT, [1, None])
    ts_in      = helper.make_tensor_value_info("train_size", TensorProto.INT64, [1])
    cm_in      = helper.make_tensor_value_info("cat_mask",   TensorProto.BOOL,  [1, H])
    d_in       = helper.make_tensor_value_info("d",          TensorProto.INT64, [1])

    logits_out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, K])
    outputs = [logits_out]
    nodes = [
        helper.make_node("Reshape", inputs=["x", "reshape_shape"], outputs=["x_2d"]),
        helper.make_node("MatMul",  inputs=["x_2d", "W"],          outputs=["logits"]),
    ]
    initializers = [
        numpy_helper.from_array(W,             name="W"),
        numpy_helper.from_array(reshape_shape, name="reshape_shape"),
    ]

    if include_borders_output:
        borders_out = helper.make_tensor_value_info("borders", TensorProto.FLOAT, [K + 1])
        outputs.append(borders_out)
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["borders"],
            value=numpy_helper.from_array(
                borders.astype(np.float32), name="borders_const"
            ),
        ))

    graph = helper.make_graph(
        nodes=nodes,
        name="test_graph",
        inputs=[x_in, y_in, ts_in, cm_in, d_in],
        outputs=outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _make_valid_borders(K: int) -> np.ndarray:
    """Non-uniform strictly-increasing borders for K bins."""
    borders = np.zeros(K + 1, dtype=np.float32)
    borders[0] = -3.0
    # Non-uniform: outer bins wider than inner bins
    widths = np.array([0.5 * (1 + abs(i - K / 2) / (K / 2)) for i in range(K)],
                      dtype=np.float32)
    for i in range(K):
        borders[i + 1] = borders[i] + widths[i]
    return borders


def _make_fake_fixture(tmpdir: pathlib.Path, borders: np.ndarray, include_borders: bool = True) -> None:
    """Write a fake fixture directory for testing."""
    K = len(borders) - 1
    H = H_FIXTURE

    graph_bytes = _make_minimal_graph(K, H, borders, include_borders_output=include_borders)
    graph_path = tmpdir / "graph_tabpfn_v2.onnx"
    graph_path.write_bytes(graph_bytes)

    manifest = {
        "manifest_version": 1,
        "model_id": "test_fixture",
        "task": "regression",
        "license": "fixture-mit",
        "repo": "local:test/fixtures/tabpfn_v2",
        "revision": "fixture-v1",
        "graph": "graph_tabpfn_v2.onnx",
        "graph_id": "test",
        "tensor_map": "tensor_map_tabpfn_v2.json",
        "preprocessing_profile": "tabpfn_v2",
        "distribution_output": True,
        "engine_profiles": {"cpu": {"dtype": "f32"}},
        "files": [],
    }
    (tmpdir / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestCommittedFixtureContract:
    """Test 1 (Positive): check_tabpfn_v2_contract passes on the committed fixture."""

    def test_committed_fixture_passes(self):
        """The committed test/fixtures/tabpfn_v2 satisfies the distribution contract."""
        if not _FIXTURE_DIR.exists():
            pytest.skip(f"Fixture not built yet: {_FIXTURE_DIR}")
        check_tabpfn_v2_contract(_FIXTURE_DIR, K=K_FIXTURE)

    def test_committed_fixture_sha256(self):
        """FIXTURE_SHA256 file matches committed artifact sha256 values."""
        import hashlib

        sha_file = _FIXTURE_DIR / "FIXTURE_SHA256"
        if not sha_file.exists():
            pytest.skip(f"FIXTURE_SHA256 not found: {sha_file}")

        lines = sha_file.read_text().strip().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split("  ", 1)
            assert len(parts) == 2, f"unexpected FIXTURE_SHA256 line format: {line!r}"
            expected_sha, name = parts
            artifact = _FIXTURE_DIR / name
            assert artifact.exists(), f"artifact listed in FIXTURE_SHA256 not found: {name}"
            actual_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
            assert actual_sha == expected_sha, (
                f"sha256 mismatch for {name}: expected {expected_sha}, got {actual_sha}"
            )


class TestNegativeBordersNotIncreasing:
    """Test 2 (Negative): non-increasing borders fail the check."""

    def test_flat_borders_fail(self):
        """Flat (non-increasing) borders should fail the contract check."""
        K = K_FIXTURE
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            # Make borders that are NOT strictly increasing (all same value)
            bad_borders = np.zeros(K + 1, dtype=np.float32)  # all 0.0 — not increasing
            _make_fake_fixture(tmpdir_path, bad_borders)
            with pytest.raises(AssertionError, match="strictly increasing"):
                check_tabpfn_v2_contract(tmpdir_path, K=K)

    def test_descending_borders_fail(self):
        """Descending borders should fail the contract check."""
        K = K_FIXTURE
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            bad_borders = np.linspace(5.0, -5.0, K + 1, dtype=np.float32)  # descending
            _make_fake_fixture(tmpdir_path, bad_borders)
            with pytest.raises(AssertionError, match="strictly increasing"):
                check_tabpfn_v2_contract(tmpdir_path, K=K)

    def test_uniform_borders_fail(self):
        """Uniform (linspace) borders should fail the non-uniformity check."""
        K = K_FIXTURE
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            # Uniform borders: technically increasing, but std of diffs ≈ 0
            uniform_borders = np.linspace(-5.0, 5.0, K + 1, dtype=np.float32)
            _make_fake_fixture(tmpdir_path, uniform_borders)
            with pytest.raises(AssertionError, match="uniform"):
                check_tabpfn_v2_contract(tmpdir_path, K=K)


class TestNegativeMissingBordersOutput:
    """Test 3 (Negative): missing 'borders' output in ONNX graph fails the check."""

    def test_missing_borders_output_fails(self):
        """Graph without 'borders' output should fail with a clear error."""
        K = K_FIXTURE
        valid_borders = _make_valid_borders(K)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            _make_fake_fixture(tmpdir_path, valid_borders, include_borders=False)
            with pytest.raises(AssertionError, match="missing 'borders' output"):
                check_tabpfn_v2_contract(tmpdir_path, K=K)

    def test_valid_fake_fixture_passes(self):
        """A properly constructed fake fixture with both outputs passes."""
        K = K_FIXTURE
        valid_borders = _make_valid_borders(K)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            _make_fake_fixture(tmpdir_path, valid_borders, include_borders=True)
            # Should not raise
            check_tabpfn_v2_contract(tmpdir_path, K=K)
