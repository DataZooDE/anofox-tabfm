"""MODL-04: Validate the tabpfn_v2 fixture's ONNX distribution output contract.

Asserts before the C++ decoder is trusted that the committed fixture ONNX graph
produces:
  - 'logits'  output: shape [T, K] float32
  - 'borders' output: shape [K+1] float32, strictly increasing

Usage:
    uv run check_tabpfn_v2             # validates test/fixtures/tabpfn_v2
    uv run check_tabpfn_v2 --fixture-dir /path/to/fixture

Exits nonzero on contract violation. Prints "PASS" on success.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort

# Repo root relative to this file: tools/parity/src/parity/check_tabpfn_v2.py
_THIS_DIR = pathlib.Path(__file__).parent
_DEFAULT_FIXTURE_DIR = _THIS_DIR.parents[3] / "test" / "fixtures" / "tabpfn_v2"

# Fixture defaults matching build_fixture.py constants
_DEFAULT_K = 16
_DEFAULT_H = 8


def check_tabpfn_v2_contract(
    fixture_dir: pathlib.Path,
    K: int = _DEFAULT_K,
    T: int = 6,
    H: int = _DEFAULT_H,
) -> None:
    """Validate the tabpfn_v2 fixture ONNX distribution contract.

    Parameters
    ----------
    fixture_dir : pathlib.Path
        Directory containing manifest.json and the fixture ONNX graph.
    K : int
        Expected number of distribution bins.
    T : int
        Number of rows to use in the validation run.
    H : int
        Feature width to use in the validation run.

    Raises
    ------
    AssertionError
        On any contract violation.
    FileNotFoundError
        If fixture_dir or manifest.json is missing.
    """
    manifest_path = fixture_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {fixture_dir}")

    manifest = json.loads(manifest_path.read_text())

    # ── Manifest contract checks ──────────────────────────────────────────────
    assert manifest.get("distribution_output") is True, (
        f"manifest.distribution_output must be true, got {manifest.get('distribution_output')!r}"
    )
    assert manifest.get("preprocessing_profile") == "tabpfn_v2", (
        f"manifest.preprocessing_profile must be 'tabpfn_v2', got {manifest.get('preprocessing_profile')!r}"
    )

    # ── Load ONNX session ─────────────────────────────────────────────────────
    graph_name = manifest.get("graph", "graph_tabpfn_v2.onnx")
    graph_path = fixture_dir / graph_name
    if not graph_path.exists():
        raise FileNotFoundError(f"ONNX graph not found: {graph_path}")

    sess = ort.InferenceSession(
        str(graph_path),
        providers=["CPUExecutionProvider"],
    )

    # ── Check output names (Pitfall 3) ────────────────────────────────────────
    output_names = {o.name for o in sess.get_outputs()}
    assert "logits" in output_names, (
        f"ONNX graph is missing 'logits' output. Got: {output_names}"
    )
    assert "borders" in output_names, (
        f"ONNX graph is missing 'borders' output. Got: {output_names}"
    )

    # ── Run with minimal inputs ───────────────────────────────────────────────
    train_size = T * 2 // 3  # 2/3 of rows are training

    feed = {
        "x":          np.zeros((1, T, H), dtype=np.float32),
        "y":          np.zeros((1, T), dtype=np.float32),
        "train_size": np.array([train_size], dtype=np.int64),
        "cat_mask":   np.zeros((1, H), dtype=bool),
        "d":          np.array([H], dtype=np.int64),
    }

    raw_outputs = sess.run(None, feed)
    out_map = {o.name: v for o, v in zip(sess.get_outputs(), raw_outputs)}

    logits = out_map["logits"]
    borders = out_map["borders"]

    # ── Shape contract ────────────────────────────────────────────────────────
    assert logits.shape == (T, K), (
        f"logits shape {logits.shape} != ({T}, {K}). "
        f"Expected [T={T}, K={K}] for tabpfn_v2 distribution output."
    )
    assert borders.shape == (K + 1,), (
        f"borders shape {borders.shape} != ({K + 1},). "
        f"Expected [K+1={K+1}] bin borders for K={K} bins."
    )

    # ── Dtype contract ────────────────────────────────────────────────────────
    assert logits.dtype == np.float32, (
        f"logits dtype {logits.dtype} != float32"
    )
    assert borders.dtype == np.float32, (
        f"borders dtype {borders.dtype} != float32"
    )

    # ── Strictly increasing borders (Pitfall 1 guard) ─────────────────────────
    diffs = np.diff(borders)
    assert np.all(diffs > 0), (
        f"borders must be strictly increasing (non-uniform). "
        f"Found non-positive diffs at indices: {np.where(diffs <= 0)[0].tolist()}. "
        f"Min diff: {float(diffs.min()):.6f}"
    )

    # ── Non-uniformity check: borders must NOT be uniform ─────────────────────
    # This guards against accidentally using np.linspace (Pitfall 1).
    diffs_std = float(np.std(diffs))
    assert diffs_std > 1e-4, (
        f"borders appear uniform (std of diffs={diffs_std:.6e}). "
        f"The real TabPFN v2 contract requires non-uniform bins. "
        f"Do not use np.linspace for border generation."
    )

    print(
        f"PASS: logits {logits.shape} float32, borders {borders.shape} float32, "
        f"strictly increasing (min_diff={float(diffs.min()):.4f}, "
        f"max_diff={float(diffs.max()):.4f}, std={diffs_std:.4f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate tabpfn_v2 fixture distribution output contract (MODL-04)."
    )
    parser.add_argument(
        "--fixture-dir",
        type=pathlib.Path,
        default=_DEFAULT_FIXTURE_DIR,
        help="Fixture directory containing manifest.json (default: test/fixtures/tabpfn_v2)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=_DEFAULT_K,
        help=f"Expected number of distribution bins (default: {_DEFAULT_K})",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=6,
        help="Number of input rows for the validation run (default: 6)",
    )
    args = parser.parse_args()

    try:
        check_tabpfn_v2_contract(
            fixture_dir=args.fixture_dir,
            K=args.K,
            T=args.T,
        )
    except (AssertionError, FileNotFoundError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
