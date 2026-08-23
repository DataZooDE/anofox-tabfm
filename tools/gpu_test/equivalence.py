#!/usr/bin/env python3
"""Cross-backend equivalence: a device switch must not change the answer.

The contract behind docs/DYNAMIC_BACKENDS.md. Every backend runs the same
graph on the same inputs and is compared against the CPU result, which is the
reference by definition — it is the one every user gets.

Tolerances are per-comparison and stated rather than assumed, because they mean
different things:

    CPU vs CPU        bit-identical   same kernels, same reduction order; any
                                      difference is a bug in a graph rewrite
    CPU vs GPU fp32   relative 1e-4   different kernels and reduction order
    GPU bf16/fp16     class agreement reduced precision is the point of the mode

Runs with synthesized initializers by default (no licensed weights), or against
a real cached checkpoint with --weights, which is the stronger statement.

    equivalence.py resources/graph_tabicl_classification.onnx --providers cpu
    equivalence.py resources/graph_tabicl_classification.onnx \\
        --providers cpu,cuda --weights ~/.cache/.../model.safetensors \\
        --tensor-map resources/tensor_map_tabicl_classification.json

A backend named "<name>-plugin" is driven through the REAL backend plugin
(tabfm_plugin_abi.h, dlopen'd from --plugin-dir) rather than through ONNX
Runtime's own execution provider. That distinction is the whole point on GPU:
since docs/DYNAMIC_BACKENDS.md phases 1 and 3, neither ROCm nor CUDA reaches
the accelerator through this process's ORT, so `--providers cpu,cuda` measures
ONNX Runtime while `--providers cpu,cuda-plugin` measures what actually ships.

    equivalence.py resources/graph_migraphx_classification.onnx \\
        --providers cpu,rocm-plugin --plugin-dir build/debug/extension/anofox_tabfm
    equivalence.py resources/graph_ext_classification.onnx \\
        --providers cpu,cuda-plugin --plugin-dir ~/.cache/anofox-tabfm/ep

The plugins only speak the 5-input tabfm signature (x, y, cat_mask, train_size,
d), so pair them with a graph_ext_* / graph_migraphx_* graph; the (x, y) graphs
are ORT-provider only. The signature is detected from the graph, not assumed.

Exit codes: 0 equivalent, 1 a backend failed to run, 2 answers diverged,
3 a requested provider was not available (never silently skipped — a GPU run
that quietly measures CPU is the failure this suite exists to prevent).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}

# The providers above are ONNX Runtime's own execution providers, reached
# through its Python bindings. Since docs/DYNAMIC_BACKENDS.md phase 1 (ROCm)
# and phase 3 (CUDA) that is NOT how this extension reaches a GPU: inference
# runs in a dlopen'd backend plugin carrying its own runtime. Comparing CPU
# against ORT's CUDA EP therefore measures ONNX Runtime, not us, and would keep
# passing while the shipped path was broken — the same class of blind spot this
# suite exists to prevent. Suffix a backend with "-plugin" to drive the real
# plugin through tabfm_plugin_abi.h instead (needs --plugin-dir).
PLUGIN_LIBS = {
    "cuda": "libanofox_tabfm_cuda_plugin.so",
    "rocm": "libanofox_tabfm_migraphx_plugin.so",
}
TABFM_PLUGIN_ABI_VERSION = 1
TABFM_PLUGIN_ENTRY_SYMBOL = "TabFMGetPluginApi"

SAFETENSORS_DTYPE = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64, "BOOL": np.bool_}
ONNX_DTYPE = {1: np.float32, 7: np.int64, 9: np.bool_, 11: np.float64}


def read_safetensors(path: Path):
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
        base = 8 + header_len
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = SAFETENSORS_DTYPE.get(meta["dtype"])
            if dtype is None:
                raise SystemExit(f"unhandled safetensors dtype {meta['dtype']}")
            begin, end = meta["data_offsets"]
            fh.seek(base + begin)
            arr = np.frombuffer(fh.read(end - begin), dtype=dtype).reshape(meta["shape"] or [1])
            if meta["dtype"] == "BF16":
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            out[name] = arr
    return out


def initializer_arrays(graph: Path, seed: int, weights: Path | None, tensor_map: Path | None):
    """name -> ndarray for every external initializer: the real checkpoint when
    given, else synthesized. Kept separate from the ORT-specific wrapping below
    because the plugin path needs the raw values to write to disk, and both
    paths must see the SAME numbers or the comparison means nothing."""
    model = onnx.load(str(graph), load_external_data=False)
    tensors = read_safetensors(weights) if weights else None
    mapping = json.loads(tensor_map.read_text())["initializers"] if tensor_map else {}
    rng = np.random.default_rng(seed)

    arrays, missing = {}, []
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        dims = list(init.dims)
        if tensors is not None:
            arr = tensors.get(mapping.get(init.name, init.name))
            if arr is None:
                missing.append(init.name)
                continue
            arr = arr.reshape(dims).astype(np.float32)
        else:
            dtype = ONNX_DTYPE.get(init.data_type)
            if dtype is None:
                raise SystemExit(f"unhandled onnx dtype {init.data_type} for {init.name}")
            arr = ((rng.standard_normal(dims) * 0.02).astype(dtype) if dtype in (np.float32, np.float64)
                   else rng.integers(0, 2, dims).astype(dtype))
        arrays[init.name] = np.ascontiguousarray(arr)
    if missing:
        raise SystemExit(f"{len(missing)} initializers unmapped, first: {missing[:3]}")
    return arrays


def initializers(graph: Path, seed: int, weights: Path | None, tensor_map: Path | None):
    """Injected initializers, wrapped as OrtValues for add_external_initializers."""
    arrays = initializer_arrays(graph, seed, weights, tensor_map)
    names = list(arrays)
    return names, [ort.OrtValue.ortvalue_from_numpy(arrays[n]) for n in names]


def materialize_external_data(graph: Path, arrays, dest: Path) -> Path:
    """Write the graph plus a real external-data file into `dest`.

    The plugin ABI takes a graph path and a weights directory — it has no way to
    receive in-memory initializers — so the values injected on the ORT side have
    to be laid down on disk at the offsets the graph declares. Returns the
    directory to hand the plugin as weights_dir.
    """
    model = onnx.load(str(graph), load_external_data=False)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(graph, dest / graph.name)

    blobs: dict[str, bytearray] = {}
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        meta = {e.key: e.value for e in init.external_data}
        location = meta["location"]
        offset, length = int(meta.get("offset", 0)), int(meta["length"])
        raw = arrays[init.name].tobytes()
        if len(raw) != length:
            raise SystemExit(f"{init.name}: graph declares {length} bytes, values are {len(raw)}")
        buf = blobs.setdefault(location, bytearray())
        if len(buf) < offset + length:
            buf.extend(b"\0" * (offset + length - len(buf)))
        buf[offset:offset + length] = raw

    for location, buf in blobs.items():
        target = dest / location
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(buf))
    return dest


# --- the plugin ABI, mirrored from src/include/tabfm_plugin_abi.h ------------
# Field order and types must match that header exactly; abi_version is checked
# at load, which is what catches a drift here rather than reading nonsense
# offsets. Bump TABFM_PLUGIN_ABI_VERSION above in step with the header.
class _CreateParams(ctypes.Structure):
    _fields_ = [("graph_path", ctypes.c_char_p), ("weights_dir", ctypes.c_char_p),
                ("cache_dir", ctypes.c_char_p), ("arch", ctypes.c_char_p),
                ("precision", ctypes.c_char_p), ("mxr_source", ctypes.c_char_p),
                ("device_ordinal", ctypes.c_int)]


class _RunInput(ctypes.Structure):
    _fields_ = [("x", ctypes.POINTER(ctypes.c_float)), ("y", ctypes.POINTER(ctypes.c_float)),
                ("cat_mask", ctypes.POINTER(ctypes.c_uint8)),
                ("t", ctypes.c_int64), ("h", ctypes.c_int64),
                ("train_size", ctypes.c_int64), ("d", ctypes.c_int64)]


class _RunOutput(ctypes.Structure):
    _fields_ = [("logits", ctypes.POINTER(ctypes.c_float)), ("logits_len", ctypes.c_int64),
                ("shape", ctypes.POINTER(ctypes.c_int64)), ("shape_len", ctypes.c_int64)]


_ERRBUF = ctypes.POINTER(ctypes.c_char)


class _Api(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int),
        ("name", ctypes.CFUNCTYPE(ctypes.c_char_p)),
        ("create", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(_CreateParams), _ERRBUF, ctypes.c_size_t)),
        ("run", ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_RunInput),
                                 ctypes.POINTER(_RunOutput), _ERRBUF, ctypes.c_size_t)),
        ("precompile", ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        _ERRBUF, ctypes.c_size_t)),
        ("free_output", ctypes.CFUNCTYPE(None, ctypes.POINTER(_RunOutput))),
        ("destroy", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]


def detect_arch(backend: str) -> str:
    """The plugin names its cache entries after the GPU arch; ask the vendor
    tool rather than guessing, and fall back to a harmless label."""
    if backend != "rocm":
        return ""
    try:
        import subprocess
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            if "gfx" in line and "Name:" in line:
                return line.split()[-1].strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def plugin_cache_dir(backend: str, weights: Path | None, workdir: Path) -> Path:
    """Where the plugin may cache compiled artifacts.

    MIGraphX compiles per shape bucket and caches the result as an .mxr keyed by
    model tag, arch, precision and bucket — deliberately NOT by the weights,
    because in production the weights for a given model never change. This tool
    breaks that assumption: with synthesized initializers the values differ from
    whatever produced a cached .mxr, and reusing it would compare the CPU run
    against a program built from entirely different numbers — a green result
    that means nothing. So synthesized runs get a throwaway cache and pay the
    compile; only a real --weights run may reuse the shared one.
    """
    if weights is None:
        return workdir / "cache"
    return Path.home() / ".cache" / "anofox-tabfm" / backend


def run_plugin(backend: str, plugin_dir: Path, graph: Path, arrays, dims, workdir: Path,
               weights: Path | None = None):
    """Drive the real backend plugin the way tabfm_plugin_backend.cpp does."""
    lib_path = plugin_dir / PLUGIN_LIBS[backend]
    if not lib_path.exists():
        raise LookupError(f"{lib_path} not found (build it, or fetch with tabfm_download_runtime('{backend}'))")
    lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_LOCAL)
    entry = getattr(lib, TABFM_PLUGIN_ENTRY_SYMBOL)
    entry.restype = ctypes.POINTER(_Api)
    api = entry().contents
    if api.abi_version != TABFM_PLUGIN_ABI_VERSION:
        raise LookupError(f"plugin ABI {api.abi_version}, expected {TABFM_PLUGIN_ABI_VERSION}")

    weights_dir = materialize_external_data(graph, arrays, workdir)
    arch = detect_arch(backend)
    err = ctypes.create_string_buffer(1024)
    cache = plugin_cache_dir(backend, weights, workdir)
    cache.mkdir(parents=True, exist_ok=True)
    params = _CreateParams(graph_path=str(weights_dir / graph.name).encode(),
                           weights_dir=str(weights_dir).encode(),
                           cache_dir=str(cache).encode(), arch=arch.encode(),
                           precision=b"fp32", mxr_source=b"", device_ordinal=0)
    handle = api.create(ctypes.byref(params), err, len(err))
    if not handle:
        raise RuntimeError(err.value.decode(errors="replace") or "create() returned NULL")
    try:
        x, y, cat_mask, T, H, S = dims
        run_input = _RunInput(x=x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              y=y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              cat_mask=cat_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                              t=T, h=H, train_size=S, d=H)
        out = _RunOutput()
        if api.run(handle, ctypes.byref(run_input), ctypes.byref(out), err, len(err)) != 0:
            raise RuntimeError(err.value.decode(errors="replace") or "run() failed")
        try:
            shape = [out.shape[i] for i in range(out.shape_len)]
            flat = np.ctypeslib.as_array(out.logits, shape=(out.logits_len,))
            return np.array(flat, dtype=np.float32).reshape(shape)  # copy before free_output
        finally:
            api.free_output(ctypes.byref(out))
    finally:
        api.destroy(handle)


def run(graph: Path, provider: str, feeds, seed, weights, tensor_map):
    ep = PROVIDERS[provider]
    names, values = initializers(graph, seed, weights, tensor_map)
    so = ort.SessionOptions()
    so.add_external_initializers(names, values)
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(graph), so, providers=[ep])
    if ep not in sess.get_providers():
        # Never fall through to CPU: a GPU comparison that silently ran on CPU
        # would "pass" while testing nothing.
        raise LookupError(f"{ep} did not instantiate (got {sess.get_providers()})")
    return sess.run(["logits"], feeds)[0]


def compare(reference, other, rel_tol):
    a = reference.astype(np.float64)
    b = other.astype(np.float64)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if np.array_equal(reference, other):
        return True, "bit-identical"
    denom = np.maximum(np.abs(a), 1e-12)
    rel = float(np.max(np.abs(a - b) / denom))
    ok = rel <= rel_tol
    detail = f"max relative {rel:.3e} (tolerance {rel_tol:.0e})"
    if reference.ndim == 3 and reference.shape[2] > 1:
        agree = float(np.mean(a.argmax(axis=2) == b.argmax(axis=2)))
        detail += f", argmax agreement {agree:.4f}"
        ok = ok and agree == 1.0
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph", type=Path)
    ap.add_argument("--providers", default="cpu", help="comma-separated: cpu,cuda,rocm,migraphx,coreml")
    ap.add_argument("--shapes", default="70x3x60,128x8x100")
    ap.add_argument("--weights", type=Path, help="real checkpoint (.safetensors) instead of synthesized values")
    ap.add_argument("--tensor-map", type=Path, help="onnx-name -> safetensors-key map for --weights")
    ap.add_argument("--rel-tol", type=float, default=1e-4, help="CPU-vs-accelerator tolerance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plugin-dir", type=Path,
                    help="directory holding the backend plugins, for the '-plugin' providers")
    args = ap.parse_args()

    wanted = [p.strip() for p in args.providers.split(",") if p.strip()]
    if "cpu" not in wanted:
        wanted.insert(0, "cpu")  # CPU is the reference by definition
    for p in wanted:
        base = p[:-len("-plugin")] if p.endswith("-plugin") else p
        if p.endswith("-plugin"):
            if base not in PLUGIN_LIBS:
                raise SystemExit(f"no plugin known for '{base}' (have: {', '.join(PLUGIN_LIBS)})")
            if not args.plugin_dir:
                raise SystemExit(f"'{p}' needs --plugin-dir")
        elif base not in PROVIDERS:
            raise SystemExit(f"unknown provider '{p}'")

    model = onnx.load(str(args.graph), load_external_data=False)
    inputs = {i.name for i in model.graph.input}
    tabfm_signature = {"cat_mask", "train_size"} <= inputs
    print(f"onnxruntime {ort.__version__}  graph {args.graph.name}")
    print(f"signature={'tabfm (x,y,cat_mask,train_size,d)' if tabfm_signature else 'x,y'}")
    print(f"reference=cpu  compare={[p for p in wanted if p != 'cpu'] or '(none — cpu self-check only)'}")
    print(f"weights={'real checkpoint' if args.weights else 'synthesized'}")
    if any(p.endswith("-plugin") for p in wanted) and not tabfm_signature:
        raise SystemExit("the backend plugins only speak the 5-input tabfm signature; "
                         "use a graph_ext_*/graph_migraphx_* graph for '-plugin' providers")

    failures = divergences = unavailable = 0
    for chunk in args.shapes.split(","):
        T, H, S = (int(v) for v in chunk.lower().split("x"))
        rng = np.random.default_rng(1234)
        x = rng.standard_normal((1, T, H)).astype(np.float32)
        if tabfm_signature:
            # y carries a label per row, with -100 marking the query rows the
            # model must predict — the same sentinel the C++ engine uses.
            y = np.full((1, T), -100.0, dtype=np.float32)
            y[0, :S] = rng.integers(0, 3, S).astype(np.float32)
            cat_mask = np.zeros((1, H), dtype=np.bool_)
            feeds = {"x": x, "y": y, "cat_mask": cat_mask,
                     "train_size": np.array([S], dtype=np.int64),
                     "d": np.array([H], dtype=np.int64)}
            plugin_dims = (np.ascontiguousarray(x.reshape(-1)),
                           np.ascontiguousarray(y.reshape(-1)),
                           np.ascontiguousarray(cat_mask.reshape(-1).astype(np.uint8)), T, H, S)
        else:
            feeds = {"x": x, "y": rng.integers(0, 3, (1, S)).astype(np.float32)}
            plugin_dims = None
        print(f"\n=== T={T} H={H} S={S}")

        results = {}
        for provider in wanted:
            try:
                if provider.endswith("-plugin"):
                    backend = provider[:-len("-plugin")]
                    arrays = initializer_arrays(args.graph, args.seed, args.weights, args.tensor_map)
                    with tempfile.TemporaryDirectory(prefix="tabfm-equiv-") as tmp:
                        results[provider] = run_plugin(backend, args.plugin_dir, args.graph, arrays,
                                                       plugin_dims, Path(tmp), args.weights)
                else:
                    results[provider] = run(args.graph, provider, feeds, args.seed, args.weights, args.tensor_map)
                print(f"  {provider:13} ran, logits {tuple(results[provider].shape)}")
            except LookupError as exc:
                print(f"  {provider:13} UNAVAILABLE: {exc}")
                unavailable += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  {provider:13} FAILED: {str(exc).strip().splitlines()[-1][:140]}")
                failures += 1

        reference = results.get("cpu")
        if reference is None:
            continue
        for provider, out in results.items():
            if provider == "cpu":
                continue
            ok, detail = compare(reference, out, args.rel_tol)
            print(f"  cpu vs {provider:13} {'OK  ' if ok else 'DIVERGED'} {detail}")
            if not ok:
                divergences += 1

    print(f"\nfailures={failures} divergences={divergences} unavailable={unavailable}")
    if failures:
        return 1
    if divergences:
        return 2
    return 3 if unavailable else 0


if __name__ == "__main__":
    sys.exit(main())
