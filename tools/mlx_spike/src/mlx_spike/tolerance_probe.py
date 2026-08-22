"""Does equivalence.py's 1e-4 relative bar do any work for CUDA/ROCm today?

S-M2 found that `tools/gpu_test/equivalence.py`'s metric -- max per-element
`|a-b| / max(|a|, 1e-12)` on LOGITS -- is failed by the torch reference
implementation against its own ONNX export. That metric is what the CUDA and
ROCm comparisons are judged by, and those are reported as passing. Both cannot
be straightforwardly true, so this measures which it is.

Method: import equivalence.py itself and use ITS synthesis, ITS shapes, ITS
compare(). Reimplementing them would test a lookalike, not the thing shipping.

The hypothesis under test is that the metric's verdict is decided by
`min |logit|` in the comparison -- a property of the weights and shape, not of
the backend -- because the denominator is floored at 1e-12. If so, the bar
passes or fails by luck of the draw, and equivalence.py's default synthesized
weights (`standard_normal * 0.02`, so a near-degenerate untrained network) sit
somewhere on that lottery that real weights do not.

Run:  uv run python -m mlx_spike.tolerance_probe
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from .common import MITRA_CACHE, REPO_ROOT, RESOURCES, softmax

SHAPES = [(70, 3, 60), (128, 8, 100)]  # equivalence.py's --shapes default


def _mark(key: str, value: object) -> None:
    print(f"TOL_{key}={value}", flush=True)


def _load_equivalence():
    """Import the module under investigation, rather than a copy of it."""
    path = REPO_ROOT / "tools" / "gpu_test" / "equivalence.py"
    spec = importlib.util.spec_from_file_location("equivalence", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["equivalence"] = mod
    spec.loader.exec_module(mod)
    return mod


def _feeds(t: int, h: int, s: int):
    """equivalence.py's own feed construction, for the 5-input tabfm signature
    minus cat_mask (the ext graphs take 4)."""
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((1, t, h)).astype(np.float32)
    y = np.full((1, t), -100.0, dtype=np.float32)
    y[0, :s] = rng.integers(0, 3, s).astype(np.float32)
    return {"x": x, "y": y,
            "train_size": np.array([s], dtype=np.int64),
            "d": np.array([h], dtype=np.int64)}, s


def _characterize(name: str, logits: np.ndarray, train_size: int) -> float:
    """Describe the logit distribution. Reported for context only -- the verdict
    comes from the measured torch-vs-ORT comparison below, not from this.

    (An earlier version of this probe extrapolated a verdict from
    `eps / min|logit|` for a fixed eps. That is wrong: fp32 error scales WITH
    the values, so a network whose logits are 1000x smaller also has ~1000x
    smaller absolute error, and the ratio need not move at all. The only honest
    way to know is to run a second implementation, which is what happens below.)
    """
    q = logits[:, train_size:, :]
    min_abs = float(np.abs(q).min())
    _mark(f"{name}_LOGIT_RANGE", f"[{q.min():.4f},{q.max():.4f}]")
    _mark(f"{name}_MIN_ABS_LOGIT", f"{min_abs:.3e}")
    return min_abs


def main() -> int:
    print("=== tolerance probe: what decides equivalence.py's verdict? ===", flush=True)
    eq = _load_equivalence()
    _mark("EQUIVALENCE_DEFAULT_RTOL", eq.__doc__.count("1e-4") and "1e-4")

    graph = RESOURCES / "graph_ext_mitra_classification.onnx"
    tensor_map = RESOURCES / "tensor_map_mitra_classification.json"
    real_weights = MITRA_CACHE["classification"] / "model.safetensors"

    sources = [("SYNTH", None, None), ("REAL", real_weights, tensor_map)]
    summary: dict[str, float] = {}

    for label, weights, tmap in sources:
        for (t, h, s) in SHAPES:
            feeds, train_size = _feeds(t, h, s)
            tag = f"{label}_{t}x{h}x{s}"
            try:
                logits = eq.run(graph, "cpu", feeds, 0, weights, tmap)
            except Exception as exc:  # noqa: BLE001 - report, do not mask
                _mark(f"{tag}_ERROR", repr(exc)[:120])
                continue
            summary[tag] = _characterize(tag, logits, train_size)

    # --- the measurement: a genuine second implementation, both weight sources -
    # torch is the definition (the ONNX graph was exported from it); ORT is the
    # reference. Both are CPU, so if the strict metric fails HERE it cannot be
    # measuring backend quality -- and it is the same metric CUDA and ROCm are
    # judged by. Run for SYNTH too, because synthesized weights are what
    # equivalence.py uses when --weights is omitted, i.e. its default mode.
    _mark("NOTE", "torch-vs-ORT below is a CPU/CPU pair: no accelerator involved")
    verdicts = {}
    for label, weights, tmap in sources:
        for (t, h, s) in SHAPES:
            feeds, train_size = _feeds(t, h, s)
            ort_logits = eq.run(graph, "cpu", feeds, 0, weights, tmap)
            torch_logits = _torch_forward(eq, graph, feeds, weights, tmap)
            a = ort_logits[:, train_size:, :]
            b = torch_logits[:, train_size:, :]
            ok, detail = eq.compare(a, b, 1e-4)
            prob = float(np.abs(softmax(a) - softmax(b)).max())
            tag = f"{label}_{t}x{h}x{s}"
            _mark(f"{tag}_TORCH_VS_ORT",
                  f"{'PASS' if ok else 'FAIL'} under equivalence.compare: {detail}")
            _mark(f"{tag}_TORCH_VS_ORT_LOGIT_MAX_ABS", f"{float(np.abs(a - b).max()):.3e}")
            _mark(f"{tag}_TORCH_VS_ORT_PROB_ABS", f"{prob:.3e}")
            verdicts[tag] = ok

    real_ok = all(v for k, v in verdicts.items() if k.startswith("REAL"))
    synth_ok = all(v for k, v in verdicts.items() if k.startswith("SYNTH"))
    _mark("BAR_HOLDS_WITH_REAL_WEIGHTS", real_ok)
    _mark("BAR_HOLDS_WITH_SYNTHESIZED_WEIGHTS", synth_ok)
    _mark("RESULT", "PASS")
    return 0


def _torch_forward(eq, graph: Path, feeds, weights, tmap) -> np.ndarray:
    """Run the definition. With `weights=None` the state dict is built from
    equivalence.py's OWN synthesized initializers, so both sides of the
    comparison see identical numbers -- otherwise the comparison is meaningless."""
    src = REPO_ROOT / "tools" / "export_mitra" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import torch
    from safetensors.torch import load_file

    from export_mitra.export import ExportWrapper, _norm_name
    from export_mitra.mitra_model_patched import Tab2D

    if weights is not None:
        state = load_file(str(weights))
    else:
        arrays = eq.initializer_arrays(graph, 0, None, None)
        state = {_norm_name(k): torch.from_numpy(np.asarray(v)) for k, v in arrays.items()}

    model = Tab2D(dim=512, dim_output=10, n_layers=12, n_heads=4, task="CLASSIFICATION")
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return ExportWrapper(model).eval()(
            torch.from_numpy(feeds["x"]), torch.from_numpy(feeds["y"]),
            torch.from_numpy(feeds["train_size"]), torch.from_numpy(feeds["d"]),
        ).numpy()


if __name__ == "__main__":
    raise SystemExit(main())
