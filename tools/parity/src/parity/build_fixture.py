"""MODL-01: Build the K=16 weight-free tabpfn_v2 fixture family.

Generates the committed test/fixtures/tabpfn_v2/ artifacts:
  - graph_tabpfn_v2.onnx  (weight-free after stripping)
  - model.safetensors      (random-init, single __metadata__ key for sha256 determinism)
  - tensor_map_tabpfn_v2.json
  - manifest.json          (distribution_output=true, preprocessing_profile='tabpfn_v2')
  - golden.json            (logits+borders+decoded mean/quantiles for plan 04 C++ test)
  - FIXTURE_SHA256         (one-line-per-file sha256 for CI pinning)

License wall: ZERO Google/TabPFN weight bytes. All weights are seeded random-init (numpy
default_rng(42)) — the fixture exercises the engine's distribution decode path, not real
TabPFN v2 inference.

Double-build determinism is asserted before writing FIXTURE_SHA256.

Usage:
    uv run build_tabpfn_v2_fixture [--out <dir>]
    # default out: <repo_root>/test/fixtures/tabpfn_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors.numpy import save_file as st_save_file

# ── Fixture constants ──────────────────────────────────────────────────────────
SEED = 42           # numpy RNG seed — NEVER change; sha256 is pinned
K = 16              # number of distribution bins (small for CI speed)
H = 8               # feature width (fixture graph input)
WEIGHT_SCALE = 0.1  # scale for random-init linear layer weights

# Repo root relative to this file: tools/parity/src/parity/build_fixture.py
_THIS_DIR = pathlib.Path(__file__).parent
_DEFAULT_OUT = _THIS_DIR.parents[3] / "test" / "fixtures" / "tabpfn_v2"

# Fixture files included in FIXTURE_SHA256 and manifest.files[]
FIXTURE_FILES = [
    "model.safetensors",
    "graph_tabpfn_v2.onnx",
    "tensor_map_tabpfn_v2.json",
    "golden.json",
    "manifest.json",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sorted_borders(rng: np.random.Generator, k: int) -> np.ndarray:
    """Generate K+1 strictly-increasing non-uniform border values.

    Designed to mimic the real TabPFN v2 border properties:
      - Non-uniform spacing (outer bins wider than inner bins)
      - Strictly increasing
      - Range roughly [-5, +5] in z-space

    Uses a mix of small central increments and large outer increments
    to produce the non-uniformity property required by the contract.
    """
    # Start with base points: tight central region, wide outer region
    # widths array (K values): alternating pattern of narrow/wide
    central = K // 2
    widths = np.zeros(k, dtype=np.float32)
    for i in range(k):
        dist_from_center = abs(i - (k - 1) / 2.0)
        # Exponentially wider bins toward the tails
        widths[i] = 0.3 * math.exp(0.15 * dist_from_center)
    # Add small noise (seeded) so the pattern is not perfectly symmetric
    noise = (rng.random(k).astype(np.float32) - 0.5) * 0.05
    widths = widths + noise
    widths = np.maximum(widths, 0.01)  # strictly positive

    # Build borders from widths: b[0] anchored, each b[i+1] = b[i] + widths[i]
    total_width = widths.sum()
    b0 = -total_width / 2.0
    borders = np.zeros(k + 1, dtype=np.float32)
    borders[0] = b0
    for i in range(k):
        borders[i + 1] = borders[i] + widths[i]

    # Verify strictly increasing
    assert np.all(np.diff(borders) > 0), "borders must be strictly increasing"
    return borders


# ── ONNX graph construction ────────────────────────────────────────────────────

def _build_onnx_graph(
    W: np.ndarray,   # [H, K] linear layer weights (used only for golden run, not embedded)
    b: np.ndarray,   # [K] bias (same)
    borders_vals: np.ndarray,  # [K+1] constant borders
) -> onnx.ModelProto:
    """Build the K=16 tabpfn_v2 fixture ONNX graph with external-data stubs for W and b.

    W and b are EXTERNAL-DATA STUBS (data_location=EXTERNAL, raw_data empty) so the
    C++ engine can inject them via ORT AddExternalInitializers (requires external-data
    format). The graph is already 'weight-free' as committed — no strip step needed.

    Input convention matches the existing engine (see ORT Run in tabfm_ort_engine.cpp):
      x          float32 [1, T, H]   — features (batch dim 1, T rows, H cols)
      y          float32 [1, T]      — target (unused by this fixture graph)
      train_size int64   [1]         — number of train rows (unused)
      cat_mask   bool    [1, H]      — categorical mask (unused)
      d          int64   [1]         — active feature count (unused)

    Internally: reshape x to [T, H], matmul with W [H, K], add b [K] → logits [T, K]

    Outputs (EXPLICITLY NAMED per Pitfall 3/7):
      logits   float32 [T, K]   — unnormalized logits per row
      borders  float32 [K+1]    — constant bin borders (non-uniform, strictly increasing)
    """
    K_ = W.shape[1]   # K=16
    H_ = W.shape[0]   # H=8

    # ── Graph inputs ──────────────────────────────────────────────────────────
    x_in       = helper.make_tensor_value_info("x",          TensorProto.FLOAT, [1, None, H_])
    y_in       = helper.make_tensor_value_info("y",          TensorProto.FLOAT, [1, None])
    ts_in      = helper.make_tensor_value_info("train_size", TensorProto.INT64, [1])
    cm_in      = helper.make_tensor_value_info("cat_mask",   TensorProto.BOOL,  [1, H_])
    d_in       = helper.make_tensor_value_info("d",          TensorProto.INT64, [1])

    # ── Graph outputs (explicitly named — Pitfall 3) ─────────────────────────
    logits_out  = helper.make_tensor_value_info("logits",  TensorProto.FLOAT, [None, K_])
    borders_out = helper.make_tensor_value_info("borders", TensorProto.FLOAT, [K_ + 1])

    # ── ONNX initializers (W, b, reshape_shape) ───────────────────────────────
    # W and b are EXTERNAL-DATA STUBS so the C++ engine can inject them via
    # ORT's AddExternalInitializers API (which requires data_location=EXTERNAL).
    # The stub format: dims declared, raw_data empty, data_location=EXTERNAL,
    # external_data[location] = 'model.safetensors' (documentation only — the
    # location is ignored when injecting via AddExternalInitializers; the C++
    # engine reads the actual weights from the safetensors file by name match).
    W_init = onnx.TensorProto()
    W_init.name = "W"
    W_init.data_type = TensorProto.FLOAT
    W_init.dims.extend([H_, K_])
    W_init.data_location = TensorProto.EXTERNAL
    W_init.external_data.add().CopyFrom(
        onnx.StringStringEntryProto(key="location", value="model.safetensors")
    )
    b_init = onnx.TensorProto()
    b_init.name = "b"
    b_init.data_type = TensorProto.FLOAT
    b_init.dims.extend([K_])
    b_init.data_location = TensorProto.EXTERNAL
    b_init.external_data.add().CopyFrom(
        onnx.StringStringEntryProto(key="location", value="model.safetensors")
    )
    # Shape for Reshape: [-1, H] so dynamic T is handled — structural constant,
    # not a model weight, stays as regular initializer.
    shape_vals = np.array([-1, H_], dtype=np.int64)
    shape_init = numpy_helper.from_array(shape_vals, name="reshape_shape")

    # ── Nodes ──────────────────────────────────────────────────────────────────
    # 1. Squeeze batch dim: x [1,T,H] → x_2d [T,H] via Reshape
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["x", "reshape_shape"],
        outputs=["x_2d"],
    )

    # 2. MatMul: x_2d [T,H] @ W [H,K] → mm_out [T,K]
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["x_2d", "W"],
        outputs=["mm_out"],
    )

    # 3. Add bias: mm_out [T,K] + b [K] → logits [T,K]
    add_node = helper.make_node(
        "Add",
        inputs=["mm_out", "b"],
        outputs=["logits"],
    )

    # 4. Constant node for borders [K+1] — output explicitly named "borders"
    #    (Pitfall 7: must be wired into graph.output)
    borders_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["borders"],
        value=numpy_helper.from_array(
            borders_vals.astype(np.float32), name="borders_const"
        ),
    )

    # ── Assemble graph ────────────────────────────────────────────────────────
    graph = helper.make_graph(
        nodes=[reshape_node, matmul_node, add_node, borders_node],
        name="tabpfn_v2_fixture",
        inputs=[x_in, y_in, ts_in, cm_in, d_in],
        outputs=[logits_out, borders_out],     # both outputs listed (Pitfall 7)
        initializer=[W_init, b_init, shape_init],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 8
    model.doc_string = (
        "anofox-tabfm K=16 tabpfn_v2 fixture — random-init, weight-free after strip. "
        "License: fixture-mit. Zero Google/TabPFN weight bytes (license wall). "
        "MODL-01 fixture; plan 02-03."
    )

    # Skip onnx.checker: external-data stubs with no data file cause checker to fail.
    # The graph structure is correct; the parity test validates it at runtime via ORT
    # injection (tools/parity/tests/test_check_tabpfn_v2.py).
    return model


# ── Weight-free stripping ──────────────────────────────────────────────────────

def strip_weights(model: onnx.ModelProto) -> onnx.ModelProto:
    """Zero out model weight initializers (W, b), preserving structural constants.

    Structural constants (reshape_shape etc.) must remain intact for the graph
    to run after stripping — the parity check validates the weight-free graph
    directly without re-injecting weights (Pitfall: reshape_shape [-1, H] must
    stay so Reshape nodes produce the correct output shape).

    Only the model weight tensors (W, b) are zeroed; other initializers are
    preserved intact.
    """
    # Names of model weight tensors to zero out (safe to strip for the fixture)
    WEIGHT_NAMES = {"W", "b"}
    for init in model.graph.initializer:
        if init.name not in WEIGHT_NAMES:
            continue  # preserve structural constants (reshape_shape etc.)
        shape = [d for d in init.dims]
        dtype = init.data_type
        if dtype == TensorProto.FLOAT:
            init.CopyFrom(numpy_helper.from_array(
                np.zeros(shape, dtype=np.float32), name=init.name
            ))
        else:
            # Fallback: zero raw bytes
            init.raw_data = b"\x00" * len(init.raw_data)
    return model


def assert_weight_free(model: onnx.ModelProto) -> None:
    """Assert model weight initializers (W, b) carry zero data."""
    for init in model.graph.initializer:
        if init.name in ("W", "b"):
            arr = numpy_helper.to_array(init)
            assert np.all(arr == 0), f"initializer '{init.name}' is not zero after strip"


# ── Decode math (Python reference for golden.json) ────────────────────────────

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def distribution_mean(logits_row: np.ndarray, borders: np.ndarray) -> float:
    """FullSupportBarDistribution mean per SPIKE-tabpfn-v2-tensor-contract.md.

    Uses half-normal correction for outermost bins.
    """
    K_ = len(logits_row)
    probs = softmax(logits_row)
    midpoints = 0.5 * (borders[:-1] + borders[1:])
    widths = borders[1:] - borders[:-1]

    mean = 0.0
    for i in range(K_):
        if i == 0:
            # Left tail: half-normal correction
            contrib = borders[0] - math.sqrt(math.pi / 2.0) * widths[0]
        elif i == K_ - 1:
            # Right tail: half-normal correction
            contrib = borders[K_] + math.sqrt(math.pi / 2.0) * widths[K_ - 1]
        else:
            contrib = midpoints[i]
        mean += probs[i] * contrib
    return float(mean)


def distribution_quantile(logits_row: np.ndarray, borders: np.ndarray, q: float) -> float:
    """FullSupportBarDistribution quantile (icdf) via CDF search + linear interpolation."""
    probs = softmax(logits_row)
    widths = borders[1:] - borders[:-1]
    cumprobs = np.cumsum(probs)

    K_ = len(probs)
    # Find bin i where cumprobs[i-1] < q <= cumprobs[i]
    prev_cum = 0.0
    for i in range(K_):
        cum_i = cumprobs[i]
        if cum_i >= q or i == K_ - 1:
            if probs[i] > 1e-15:
                # Linear interpolation within bin
                t = (q - prev_cum) / probs[i]
                return float(borders[i] + t * widths[i])
            else:
                return float(0.5 * (borders[i] + borders[i + 1]))
        prev_cum = cum_i
    return float(borders[-1])


QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ── Main build function ────────────────────────────────────────────────────────

def build(out: pathlib.Path) -> dict:
    """Generate all tabpfn_v2 fixture artifacts into `out`. Returns {file: sha256}.

    License wall: zero Google/TabPFN weight bytes. All weights are seeded random-init.
    Double-build determinism is asserted before returning.
    """
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # ── 1. Random-init weights (seeded, deterministic) ─────────────────────────
    W = (rng.standard_normal((H, K)) * WEIGHT_SCALE).astype(np.float32)
    b = np.zeros(K, dtype=np.float32)
    borders_vals = _sorted_borders(np.random.default_rng(SEED + 1), K)

    # ── 2. Build ONNX graph with external-data stubs for W and b ──────────────
    # The graph is already "weight-free" as built (W/b are stubs with data_location=EXTERNAL
    # and no raw_data). No strip step needed; the C++ engine injects W/b from safetensors
    # via ORT AddExternalInitializers (which requires data_location=EXTERNAL).
    model_stub = _build_onnx_graph(W, b, borders_vals)
    graph_bytes_stub = model_stub.SerializeToString()

    # ── 3. Safetensors — single __metadata__ key for sha256 determinism ────────
    #    Save W and b as the "model weights" (borders are a Constant node, not safetensors)
    st_path = out / "model.safetensors"
    tensors = {"W": W, "b": b}
    st_save_file(
        tensors,
        str(st_path),
        metadata={"origin": (
            "anofox-tabfm CI fixture tabpfn_v2, random init, license fixture-mit "
            "(our weights, not Google's / TabPFN's). MODL-01 plan 02-03."
        )},
    )

    # ── 4. tensor_map: maps ONNX initializer names → safetensors keys ──────────
    tensor_map = {
        "initializers": {
            "W": "W",
            "b": "b",
        },
        "transforms": {},
        "safetensors": "model.safetensors",
        "task": "regression",
    }
    tm_path = out / "tensor_map_tabpfn_v2.json"
    tm_path.write_text(json.dumps(tensor_map, indent=2) + "\n")

    # ── 5. Write the external-data-stub graph (already weight-free) ──────────
    graph_path = out / "graph_tabpfn_v2.onnx"
    graph_path.write_bytes(graph_bytes_stub)

    # ── 6. Generate golden.json: run ORT via AddExternalInitializers ─────────
    #    (mirrors the C++ injection path: stubs + safetensors → ORT)
    import onnxruntime as ort
    from safetensors.numpy import load_file as st_load_file

    st_tensors = st_load_file(str(st_path))
    W_ort = ort.OrtValue.ortvalue_from_numpy(st_tensors["W"].astype(np.float32), "cpu")
    b_ort = ort.OrtValue.ortvalue_from_numpy(st_tensors["b"].astype(np.float32), "cpu")

    opts = ort.SessionOptions()
    opts.add_external_initializers(["W", "b"], [W_ort, b_ort])

    sess = ort.InferenceSession(
        graph_bytes_stub,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    # Golden inputs: T=6 rows, train_size=4, H=8 features
    T_GOLDEN = 6
    TRAIN_SIZE = 4
    rng_golden = np.random.default_rng(SEED + 2)
    x_golden = rng_golden.standard_normal((1, T_GOLDEN, H)).astype(np.float32)
    y_golden = rng_golden.standard_normal((1, T_GOLDEN)).astype(np.float32)
    ts_golden = np.array([TRAIN_SIZE], dtype=np.int64)
    cm_golden = np.zeros((1, H), dtype=bool)
    d_golden = np.array([H], dtype=np.int64)

    golden_feed = {
        "x":          x_golden,
        "y":          y_golden,
        "train_size": ts_golden,
        "cat_mask":   cm_golden,
        "d":          d_golden,
    }
    outputs = {o.name: v for o, v in zip(
        sess.get_outputs(), sess.run(None, golden_feed)
    )}

    golden_logits = outputs["logits"]    # [T, K]
    golden_borders = outputs["borders"]  # [K+1]

    assert golden_logits.shape == (T_GOLDEN, K), f"logits shape {golden_logits.shape}"
    assert golden_borders.shape == (K + 1,), f"borders shape {golden_borders.shape}"
    assert np.all(np.diff(golden_borders) > 0), "borders not strictly increasing"

    # Python-derived distribution decode for n_test rows (rows >= train_size)
    # y_mean / y_std from training targets (rows 0..TRAIN_SIZE-1)
    y_train = y_golden[0, :TRAIN_SIZE]
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-8:
        y_std = 1.0  # degenerate: flat training targets

    # z-space borders already in golden_borders; affine transform to raw space
    raw_borders = golden_borders * y_std + y_mean

    # For each test row compute mean and quantiles IN RAW SPACE (using raw_borders)
    n_test = T_GOLDEN - TRAIN_SIZE
    test_logits = golden_logits[TRAIN_SIZE:]  # [n_test, K]
    decoded_means = []
    decoded_quantiles = []
    for row_i in range(n_test):
        row_logits = test_logits[row_i]
        m = distribution_mean(row_logits, raw_borders)
        qs = [distribution_quantile(row_logits, raw_borders, q) for q in QUANTILE_LEVELS]
        decoded_means.append(m)
        decoded_quantiles.append(qs)

    golden_data = {
        "_doc": {
            "purpose": (
                "C++ parity: plan 04 loads this fixture via SET anofox_tabfm_model_manifest "
                "and asserts that DecodeDistribution matches these Python-derived decoded "
                "mean/quantiles for test rows (rows >= train_size)."
            ),
            "decode_math": (
                "FullSupportBarDistribution: mean via softmax @ midpoints + half-normal "
                "outer-bin correction; quantiles via CDF cumsum + linear interpolation. "
                "Borders are z-space; affine transform raw=z*y_std+y_mean applied. "
                "See SPIKE-tabpfn-v2-tensor-contract.md Decode Math."
            ),
            "rtol": 1e-4,
            "y_convention": (
                "y[0..train_size-1] are training targets; y[train_size..] are test rows "
                "(set to 0.0 as placeholder). Decode is for rows >= train_size only."
            ),
        },
        "inputs": {
            "x":          x_golden.tolist(),
            "y":          y_golden.tolist(),
            "train_size": int(TRAIN_SIZE),
            "cat_mask":   cm_golden.tolist(),
            "d":          int(H),
        },
        "logits":           golden_logits.tolist(),    # [T, K] float32
        "borders":          golden_borders.tolist(),   # [K+1] float32, z-space
        "y_mean":           y_mean,
        "y_std":            y_std,
        "raw_borders":      raw_borders.tolist(),      # [K+1] float64, raw space
        "decoded": {
            "n_test":     n_test,
            "means":      decoded_means,       # [n_test] raw-space
            "quantiles":  decoded_quantiles,   # [n_test][9] raw-space
            "levels":     QUANTILE_LEVELS,
        },
    }
    golden_path = out / "golden.json"
    golden_path.write_text(json.dumps(golden_data, indent=2) + "\n")

    # ── 7. manifest.json ─────────────────────────────────────────────────────
    def file_entry(name: str) -> dict:
        p = out / name
        return {"path": name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}

    manifest = {
        "manifest_version": 1,
        "model_id": "tabpfn_v2_fixture",
        "task": "regression",
        "license": "fixture-mit",
        "repo": "local:test/fixtures/tabpfn_v2",
        "revision": "fixture-v1",
        "graph": "graph_tabpfn_v2.onnx",
        "graph_id": "tabpfn-v2-fixture-opset18",
        "tensor_map": "tensor_map_tabpfn_v2.json",
        "preprocessing_profile": "tabpfn_v2",
        "distribution_output": True,
        "engine_profiles": {"cpu": {"dtype": "f32"}},
        "files": [
            file_entry("model.safetensors"),
            file_entry("graph_tabpfn_v2.onnx"),
            file_entry("tensor_map_tabpfn_v2.json"),
        ],
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # ── 8. Double-build determinism assertion ─────────────────────────────────
    #    Rebuild key artifacts from scratch and compare sha256
    rng2 = np.random.default_rng(SEED)
    W2 = (rng2.standard_normal((H, K)) * WEIGHT_SCALE).astype(np.float32)
    b2 = np.zeros(K, dtype=np.float32)
    borders2 = _sorted_borders(np.random.default_rng(SEED + 1), K)

    # Rebuild external-data stub graph
    model2 = _build_onnx_graph(W2, b2, borders2)
    graph_bytes2 = model2.SerializeToString()
    assert graph_bytes2 == graph_path.read_bytes(), \
        "graph_tabpfn_v2.onnx is NOT byte-deterministic on double build"

    # Rebuild safetensors
    tmp_st = out / "model.safetensors.recheck"
    st_save_file(
        {"W": W2, "b": b2},
        str(tmp_st),
        metadata={"origin": (
            "anofox-tabfm CI fixture tabpfn_v2, random init, license fixture-mit "
            "(our weights, not Google's / TabPFN's). MODL-01 plan 02-03."
        )},
    )
    st_sha_orig = sha256_file(st_path)
    st_sha2 = sha256_file(tmp_st)
    tmp_st.unlink()
    assert st_sha_orig == st_sha2, \
        f"model.safetensors is NOT byte-deterministic: {st_sha_orig} != {st_sha2}"

    # ── 9. FIXTURE_SHA256 (one line per file, for CI verification) ───────────
    hashes = {name: sha256_file(out / name) for name in FIXTURE_FILES}
    sha_lines = "\n".join(f"{hashes[n]}  {n}" for n in FIXTURE_FILES) + "\n"
    (out / "FIXTURE_SHA256").write_text(sha_lines)

    total = sum((out / name).stat().st_size for name in FIXTURE_FILES)
    assert total < 2 * 1024 * 1024, f"fixture total {total} B >= 2 MB budget"

    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the K=16 weight-free tabpfn_v2 fixture family."
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT,
        help="Output directory (default: test/fixtures/tabpfn_v2 relative to repo root)",
    )
    args = parser.parse_args()
    hashes = build(args.out)
    print(f"Built fixture to {args.out}:")
    for name, digest in hashes.items():
        print(f"  {digest}  {name}")
    print("Double-build determinism: PASS")


if __name__ == "__main__":
    main()
