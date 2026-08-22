"""S-M3 -- drive the built plugin through the real C ABI and check its answers.

`tools/gpu_test/plugin_load_check.c` already proves the artifact *loads* on this
machine (and it compiled on macOS unchanged, as the plan predicted). Loading is
not the interesting part: a plugin that loads and returns wrong logits passes
that check. This drives create/run/free_output/destroy through
`tabfm_plugin_abi.h` via ctypes and compares the result to the same ORT CPU
golden S-M2 used.

ctypes rather than a C++ host on purpose -- the struct layouts here are the ABI
contract, so declaring them independently from the C header is a second reading
of that contract. A mismatch in field order or size shows up as garbage rather
than being papered over by including the header that defined it.

Run:  uv run python -m mlx_spike.sm3_abi <path/to/plugin.dylib>
"""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import numpy as np

from .common import MITRA_CACHE, SpikeError, compare
from .sm1_env import GOLDEN
from .sm2_parity import LOGIT_ABS_TOL, PROB_TOL

ABI_VERSION = 1


class RunInput(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.POINTER(ctypes.c_float)),
        ("y", ctypes.POINTER(ctypes.c_float)),
        ("cat_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("t", ctypes.c_int64),
        ("h", ctypes.c_int64),
        ("train_size", ctypes.c_int64),
        ("d", ctypes.c_int64),
    ]


class RunOutput(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("logits_len", ctypes.c_int64),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("shape_len", ctypes.c_int64),
    ]


class CreateParams(ctypes.Structure):
    _fields_ = [
        ("graph_path", ctypes.c_char_p),
        ("weights_dir", ctypes.c_char_p),
        ("cache_dir", ctypes.c_char_p),
        ("arch", ctypes.c_char_p),
        ("precision", ctypes.c_char_p),
        ("mxr_source", ctypes.c_char_p),
        ("device_ordinal", ctypes.c_int),
    ]


class PluginApi(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int),
        ("name", ctypes.CFUNCTYPE(ctypes.c_char_p)),
        ("create", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(CreateParams),
                                    ctypes.c_char_p, ctypes.c_size_t)),
        ("run", ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(RunInput),
                                 ctypes.POINTER(RunOutput), ctypes.c_char_p, ctypes.c_size_t)),
        ("precompile", ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_char_p, ctypes.c_size_t)),
        ("free_output", ctypes.CFUNCTYPE(None, ctypes.POINTER(RunOutput))),
        ("destroy", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]


def _mark(key: str, value: object) -> None:
    print(f"SM3_{key}={value}", flush=True)


def _run_all(api, weights_dir, g, precision: str) -> dict:
    """Every golden case at `precision`, returning raw logits per case."""
    err = ctypes.create_string_buffer(1024)
    params = CreateParams(graph_path=b"", weights_dir=str(weights_dir).encode(), cache_dir=b"",
                          arch=b"mitra-classification", precision=precision.encode(), mxr_source=b"",
                          device_ordinal=0)
    handle = api.create(ctypes.byref(params), err, ctypes.sizeof(err))
    if not handle:
        raise SpikeError(f"create({precision}) failed: {err.value.decode()}")
    out = {}
    for name in sorted({k.split("__")[0] for k in g.files}):
        x = np.ascontiguousarray(g[f"{name}__x"], dtype=np.float32)
        y = np.ascontiguousarray(g[f"{name}__y"], dtype=np.float32)
        train_size, d = (int(v) for v in g[f"{name}__meta"])
        ri = RunInput(x=x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      y=y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      cat_mask=None, t=x.shape[1], h=x.shape[2], train_size=train_size, d=d)
        ro = RunOutput()
        if api.run(handle, ctypes.byref(ri), ctypes.byref(ro), err, ctypes.sizeof(err)) != 0:
            raise SpikeError(f"{name}: run({precision}) failed: {err.value.decode()}")
        shape = [ro.shape[i] for i in range(ro.shape_len)]
        out[name] = np.ctypeslib.as_array(ro.logits, shape=(ro.logits_len,)).copy().reshape(shape)
        api.free_output(ctypes.byref(ro))
    api.destroy(handle)
    return out


def _mark_unused():
    pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or len(argv) > 2:
        raise SpikeError("usage: python -m mlx_spike.sm3_abi <plugin.dylib> [precision]")
    plugin_path = Path(argv[0]).resolve()
    # Second argument drives the P2 check below: a precision mode that is
    # accepted but silently ignored is indistinguishable from one that works,
    # unless the LOGITS are compared rather than the predicted class.
    precision = argv[1] if len(argv) == 2 else "fp32"

    print("=== S-M3: the built plugin, driven through tabfm_plugin_abi.h ===", flush=True)
    lib = ctypes.CDLL(str(plugin_path))
    lib.TabFMGetPluginApi.restype = ctypes.POINTER(PluginApi)
    api = lib.TabFMGetPluginApi().contents
    _mark("NAME", api.name().decode())
    _mark("ABI_VERSION", api.abi_version)
    if api.abi_version != ABI_VERSION:
        raise SpikeError(f"abi_version {api.abi_version}, harness expects {ABI_VERSION}")

    weights_dir = MITRA_CACHE["classification"]
    err = ctypes.create_string_buffer(1024)

    # --- the refusal contract, before the happy path -------------------------
    # A plugin that computes correctly but fails open on the wrong architecture
    # would serve mitra's math over another model's weights. Assert it refuses.
    bad = CreateParams(graph_path=b"", weights_dir=str(weights_dir).encode(), cache_dir=b"",
                       arch=b"tabpfn-v2", precision=b"fp32", mxr_source=b"", device_ordinal=0)
    if api.create(ctypes.byref(bad), err, ctypes.sizeof(err)):
        raise SpikeError("plugin accepted arch='tabpfn-v2'; it implements mitra only")
    _mark("REFUSES_WRONG_ARCH", repr(err.value.decode()[:90]))

    # --- create --------------------------------------------------------------
    params = CreateParams(graph_path=b"", weights_dir=str(weights_dir).encode(), cache_dir=b"",
                          arch=b"mitra-classification", precision=precision.encode(), mxr_source=b"",
                          device_ordinal=0)
    _mark("PRECISION", precision)
    t0 = time.perf_counter()
    handle = api.create(ctypes.byref(params), err, ctypes.sizeof(err))
    if not handle:
        raise SpikeError(f"create failed: {err.value.decode()}")
    _mark("CREATE_MS", f"{(time.perf_counter() - t0) * 1000:.0f}")

    # precompile is a documented no-op for MLX; it must still answer OK.
    if api.precompile(handle, 100, 20, err, ctypes.sizeof(err)) != 0:
        raise SpikeError(f"precompile failed: {err.value.decode()}")
    _mark("PRECOMPILE_OK", True)

    g = np.load(weights_dir / GOLDEN)
    worst_prob, worst_logit, failed = 0.0, 0.0, []
    got_by_case: dict[str, np.ndarray] = {}
    for name in sorted({k.split("__")[0] for k in g.files}):
        x = np.ascontiguousarray(g[f"{name}__x"], dtype=np.float32)
        y = np.ascontiguousarray(g[f"{name}__y"], dtype=np.float32)
        train_size, d = (int(v) for v in g[f"{name}__meta"])
        reference = g[f"{name}__logits"]
        t, h = x.shape[1], x.shape[2]

        ri = RunInput(
            x=x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y=y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cat_mask=None, t=t, h=h, train_size=train_size, d=d,
        )
        ro = RunOutput()
        t0 = time.perf_counter()
        if api.run(handle, ctypes.byref(ri), ctypes.byref(ro), err, ctypes.sizeof(err)) != 0:
            raise SpikeError(f"{name}: run failed: {err.value.decode()}")
        dt = time.perf_counter() - t0

        shape = [ro.shape[i] for i in range(ro.shape_len)]
        got = np.ctypeslib.as_array(ro.logits, shape=(ro.logits_len,)).copy().reshape(shape)
        api.free_output(ctypes.byref(ro))  # plugin owns the buffers; it frees them
        got_by_case[name] = got

        c = compare(reference[:, train_size:, :], got[:, train_size:, :])
        # Tier the bar by precision, as tools/gpu_test/equivalence.py's header
        # states: fp32 is the strict lane where a device switch must not change
        # the answer, while bf16/fp16 are judged on CLASS AGREEMENT because
        # losing mantissa bits is the entire point of asking for them. Holding
        # bf16 to the fp32 tolerance would mark a correctly-working mode failed.
        if precision == "fp32":
            ok = (c.prob_max_abs <= PROB_TOL and c.max_abs <= LOGIT_ABS_TOL
                  and c.argmax_agreement == 1.0)
        else:
            ok = c.argmax_agreement == 1.0
        worst_prob = max(worst_prob, c.prob_max_abs)
        worst_logit = max(worst_logit, c.max_abs)
        if not ok:
            failed.append(name)
        _mark(name.upper(), f"{'ok' if ok else 'FAIL'} shape={shape} {c.describe()} ms={dt * 1000:.0f}")

    api.destroy(handle)
    _mark("DESTROY_OK", True)

    # P2 guard (docs/GPU_HARDENING_PLAN.md): the CUDA plugin once accepted this
    # setting and always ran fp32. Reduced precision MUST move the logits; if
    # bf16 reproduced fp32 bit-for-bit the mode would be a no-op wearing its
    # name. Compared against the fp32 run's own logits, not the ORT golden.
    if precision != "fp32":
        ref32 = _run_all(api, weights_dir, g, "fp32")
        moved = max(float(np.abs(ref32[k] - got_by_case[k]).max()) for k in got_by_case)
        _mark("LOGIT_SHIFT_VS_FP32", f"{moved:.3e}")
        _mark("PRECISION_ACTUALLY_APPLIED", moved > 0.0)
        if moved == 0.0:
            _mark("RESULT", "FAIL(precision silently ignored -- identical to fp32)")
            return 1
    _mark("BAR", "fp32: prob<=1e-4 + argmax" if precision == "fp32" else "reduced: argmax agreement only")
    _mark("WORST_PROB_ABS", f"{worst_prob:.3e}")
    _mark("WORST_LOGIT_ABS", f"{worst_logit:.3e}")
    if failed:
        _mark("RESULT", f"FAIL({','.join(failed)})")
        return 1
    _mark("RESULT", "PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
