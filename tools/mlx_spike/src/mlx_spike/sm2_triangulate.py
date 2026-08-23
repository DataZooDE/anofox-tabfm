"""S-M2 diagnostic -- is the MLX/ORT gap a porting bug or the ambient fp32 floor?

S-M2 came in at ~3e-05 absolute against the ORT CPU golden, with total argmax
agreement. That is either "the port is right and fp32 reassociation explains
the rest" or "the port has a small real bug". The two look identical from a
single comparison, so this asks a third party.

`mitra_model_patched.py` is the *definition* of the forward -- the ONNX graph
was exported from it, so torch eager is upstream of both ORT and MLX. Running
all three on the same inputs turns one ambiguous number into a triangle:

  torch<->ORT   the ONNX export + ORT's kernels, no MLX involved. This is the
                noise floor the project already ships with.
  torch<->MLX   the port's own error against the definition.
  ORT<->MLX     what S-M2 measures.

If torch<->MLX is comparable to torch<->ORT, the port is as faithful as the
graph is, and the S-M2 tolerance -- not the port -- is what needs fixing.

Run:  uv run python -m mlx_spike.sm2_triangulate
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .common import MITRA_CACHE, REPO_ROOT, SpikeError, compare
from .mitra_mlx import MitraMLX, Weights
from .sm1_env import GOLDEN
from .sm2_parity import PROB_TOL


def _mark(key: str, value: object) -> None:
    print(f"SM2T_{key}={value}", flush=True)


def _load_torch_reference():
    """The patched Tab2D + its ONNX ExportWrapper, loaded with the real weights."""
    src = REPO_ROOT / "tools" / "export_mitra" / "src"
    if not (src / "export_mitra" / "mitra_model_patched.py").exists():
        raise SpikeError(f"export_mitra sources not found under {src}")
    sys.path.insert(0, str(src))
    import torch
    from safetensors.torch import load_file

    from export_mitra.export import ExportWrapper
    from export_mitra.mitra_model_patched import Tab2D

    model = Tab2D(dim=512, dim_output=10, n_layers=12, n_heads=4, task="CLASSIFICATION")
    state = load_file(str(MITRA_CACHE["classification"] / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=True), None
    model.eval()
    return torch, ExportWrapper(model).eval()


def main() -> int:
    print("=== S-M2 diagnostic: torch (authority) vs ORT vs MLX ===", flush=True)
    golden_path = MITRA_CACHE["classification"] / GOLDEN
    if not golden_path.exists():
        raise SpikeError(f"golden missing at {golden_path}; run `uv run sm1_env` first")
    g = np.load(golden_path)

    torch, wrapper = _load_torch_reference()
    w = Weights(MITRA_CACHE["classification"] / "model.safetensors")
    mlx_model = MitraMLX(w, task="classification", n_heads=4, fast_sdpa=False)

    worst_ratio, strict_worst, prob_worst = 0.0, 0.0, 0.0
    for name in sorted({k.split("__")[0] for k in g.files}):
        x_np = g[f"{name}__x"]
        y_np = g[f"{name}__y"]
        train_size, d = (int(v) for v in g[f"{name}__meta"])
        ort_logits = g[f"{name}__logits"]

        with torch.no_grad():
            t0 = time.perf_counter()
            torch_logits = wrapper(
                torch.from_numpy(x_np), torch.from_numpy(y_np),
                torch.tensor([train_size], dtype=torch.int64),
                torch.tensor([d], dtype=torch.int64),
            ).numpy()
            torch_ms = (time.perf_counter() - t0) * 1000

        out = mlx_model(mx.array(x_np), mx.array(y_np), train_size, d)
        mx.eval(out)
        mlx_logits = np.asarray(out)

        q = slice(train_size, None)
        t_o = compare(torch_logits[:, q, :], ort_logits[:, q, :])
        t_m = compare(torch_logits[:, q, :], mlx_logits[:, q, :])
        o_m = compare(ort_logits[:, q, :], mlx_logits[:, q, :])
        _mark(f"{name.upper()}_TORCH_VS_ORT", t_o.describe())
        _mark(f"{name.upper()}_TORCH_VS_MLX", t_m.describe())
        _mark(f"{name.upper()}_ORT_VS_MLX", o_m.describe())
        # The verdict per case: is MLX further from the definition than ORT is?
        ratio = t_m.max_abs / max(t_o.max_abs, 1e-12)
        _mark(f"{name.upper()}_MLX_ERR_OVER_ORT_ERR", f"{ratio:.2f}x (torch fwd {torch_ms:.0f}ms)")
        worst_ratio = max(worst_ratio, ratio)

        # The evidence that retires the max-relative-on-logits metric: the
        # shipped CPU graph, judged against the code it was exported from.
        _mark(f"{name.upper()}_TORCH_VS_ORT_STRICT_REL", f"{t_o.max_rel_strict:.3e}")
        strict_worst = max(strict_worst, t_o.max_rel_strict)
        prob_worst = max(prob_worst, t_o.prob_max_abs)

    _mark("WORST_MLX_ERR_OVER_ORT_ERR", f"{worst_ratio:.2f}x")
    _mark("REFERENCE_SELF_STRICT_REL", f"{strict_worst:.3e}")
    _mark("REFERENCE_SELF_PROB_ABS", f"{prob_worst:.3e}")
    # Two claims sm2_parity's tolerance rests on, checked rather than asserted.
    _mark("CLAIM_STRICT_METRIC_FAILS_ON_REFERENCE", strict_worst > 1e-4)
    _mark("CLAIM_PROB_METRIC_PASSES_ON_REFERENCE", prob_worst <= PROB_TOL)
    _mark("CLAIM_MLX_NO_WORSE_THAN_ORT", worst_ratio <= 1.0)
    ok = strict_worst > 1e-4 and prob_worst <= PROB_TOL and worst_ratio <= 1.0
    _mark("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
