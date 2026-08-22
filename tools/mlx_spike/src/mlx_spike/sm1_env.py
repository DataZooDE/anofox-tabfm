"""S-M1 -- environment + mlx-onnx feasibility.

The plan gave this spike two jobs: decide whether route 1 (`mlx-onnx`) gives us
a second golden reference, and confirm MLX runs these graphs at all. The first
job collapsed on inspection (see the ROUTE1 verdict below), so what remains is
to establish the reference every later spike compares against, and to prove
Metal actually computes rather than falling back.

Run:  uv run sm1_env
"""

from __future__ import annotations

import importlib.util
import platform
import subprocess
import time

import numpy as np

from .common import MITRA_DIMS, SpikeError, load_mitra_weights, make_workload

GOLDEN = "golden_mitra_classification.npz"


def _mark(key: str, value: object) -> None:
    print(f"SM1_{key}={value}", flush=True)


def _host() -> None:
    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
    ).stdout.strip()
    mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
    _mark("HOST", f"{platform.system()}/{platform.machine()} {chip} {int(mem) // 2**30}GB")


def _route1_verdict() -> None:
    """Route 1 was 'try mlx-onnx first'. It does not exist.

    - github.com/ml-explore/mlx-onnx is a placeholder: one branch, three blobs
      (.gitignore, LICENSE, a 67-byte README), no code, untouched since
      2024-02-21.
    - The `mlx-onnx` name on PyPI is skryl/mlx-onnx, which exports MLX graphs
      *to* ONNX -- the opposite direction, and no use to us.
    - A GitHub search for any ONNX->MLX importer returns nothing.

    So there is no Python route to a second opinion on these graphs, and no
    op-coverage signal to harvest from one. Route 2 (hand-port) is not merely
    the product path, it is the only path; route 3 (an ONNX interpreter over
    MLX ops) remains the fallback it always was.
    """
    available = importlib.util.find_spec("mlx_onnx") is not None
    _mark("ROUTE1_IMPORTER_AVAILABLE", available)
    _mark("ROUTE1_VERDICT", "dead-no-onnx-to-mlx-importer-exists")


def _metal_proof() -> None:
    """Prove MLX computes on the GPU. `mx.default_device()` reporting Device(gpu)
    is a claim about configuration; a matmul whose result is correct AND whose
    array carries the gpu device is a claim about execution."""
    import mlx.core as mx

    _mark("MLX_VERSION", mx.__version__)
    _mark("MLX_DEFAULT_DEVICE", mx.default_device())
    _mark("MLX_METAL_AVAILABLE", mx.metal.is_available())
    if not mx.metal.is_available():
        raise SpikeError("Metal is unavailable; MLX would run on CPU and prove nothing")

    rng = np.random.default_rng(7)
    a_np = rng.standard_normal((512, 512)).astype(np.float32)
    b_np = rng.standard_normal((512, 512)).astype(np.float32)
    with mx.stream(mx.gpu):
        c = mx.matmul(mx.array(a_np), mx.array(b_np))
        mx.eval(c)
    err = float(np.abs(np.asarray(c) - (a_np @ b_np)).max())
    _mark("MLX_GPU_MATMUL_MAXERR", f"{err:.3e}")
    if err > 1e-2:  # fp32 GEMM over K=512 accumulates; this is a sanity bound
        raise SpikeError(f"MLX GPU matmul disagrees with numpy by {err}")
    # The number that decides whether M2 (tabfm-v1, 6.6 GB of weights) is
    # viable here is not hw.memsize but what Metal will let one process hold.
    info = mx.device_info()
    _mark("MLX_GPU_ARCH", info["architecture"])
    _mark("MLX_GPU_WORKING_SET_GB", f"{info['max_recommended_working_set_size'] / 2**30:.1f}")
    _mark("MLX_GPU_MAX_BUFFER_GB", f"{info['max_buffer_length'] / 2**30:.1f}")


def _weights_proof() -> None:
    st = load_mitra_weights("classification")
    _mark("WEIGHTS_PATH", st.path)
    _mark("WEIGHTS_TENSORS", len(st.names))
    _mark("WEIGHTS_HEADER_SHA_MATCHES_CPP", True)  # load_mitra_weights raises otherwise
    n_layers = 1 + max(
        int(n.split(".")[1]) for n in st.names if n.startswith("layers.")
    )
    dim = st.get("final_layer_norm.weight").shape[0]
    _mark("WEIGHTS_N_LAYERS", n_layers)
    _mark("WEIGHTS_DIM", dim)
    if n_layers != MITRA_DIMS["n_layers"] or dim != MITRA_DIMS["dim"]:
        raise SpikeError(f"weights are {n_layers}x{dim}, expected {MITRA_DIMS}")


def _golden_reference() -> None:
    from .ort_ref import OrtReference

    ref = OrtReference("classification")
    _mark("ORT_SERVED_BY", ref.served_by)
    _mark("ORT_GRAPH", ref.graph_path)

    cases = {
        # (t, h, train_size, d) -- the fixture-sized shape the plan names, a
        # padded-features case, and a larger one that exercises the quantile path.
        "small": (40, 8, 24, 8),
        "padded": (50, 16, 30, 10),
        "wide": (200, 20, 120, 20),
    }
    out: dict[str, np.ndarray] = {}
    for name, (t, h, train_size, d) in cases.items():
        w = make_workload(t=t, h=h, train_size=train_size, d=d, seed=abs(hash(name)) % 1000)
        t0 = time.perf_counter()
        logits = ref.run(w)
        dt = time.perf_counter() - t0
        out[f"{name}__x"] = w.x
        out[f"{name}__y"] = w.y
        out[f"{name}__meta"] = np.array([w.train_size, w.d], dtype=np.int64)
        out[f"{name}__logits"] = logits
        _mark(f"GOLDEN_{name.upper()}", f"shape={logits.shape} ms={dt * 1000:.1f} "
                                        f"logit_range=[{logits.min():.4f},{logits.max():.4f}]")
        # A forward that returns the same logits for every row would "match"
        # any equally-broken candidate. Assert the reference is discriminating.
        query = logits[0, train_size:, :]
        spread = float(query.max(axis=-1).mean() - query.mean())
        _mark(f"GOLDEN_{name.upper()}_MARGIN", f"{spread:.4f}")
        if spread < 1e-3:
            raise SpikeError(f"{name}: reference logits are undifferentiated; parity would be vacuous")

    path = ref.graph_path.parent / GOLDEN
    np.savez(path, **out)
    _mark("GOLDEN_SAVED", path)


def main() -> int:
    print("=== S-M1: environment + route-1 feasibility (docs/MLX_PLAN.md) ===", flush=True)
    _host()
    _route1_verdict()
    _metal_proof()
    _weights_proof()
    _golden_reference()
    _mark("RESULT", "PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
