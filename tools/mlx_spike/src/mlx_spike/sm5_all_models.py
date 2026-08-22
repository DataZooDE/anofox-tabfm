"""S-M5 — does the ONNX-over-MLX interpreter run EVERY shipped graph correctly?

Op coverage is not correctness. `onnx_mlx.HANDLERS` covers all 64 op kinds
across the 13 `graph_ext_*` graphs, but that only says nothing raises
NotImplementedError; it says nothing about whether `Pad` packs its widths the
way ONNX specifies or whether `ScatterND` flattens the right axes. This runs
each graph on both ORT CPU and the interpreter and compares.

Weights are SYNTHESIZED by default, and that is deliberate rather than a
shortcut: what is under test is the interpreter's op semantics and dynamic
shape handling, and for that both sides only need the *same* numbers. It also
means every model is testable without downloading 7 checkpoints or accepting
their licences — including tabfm-v1, whose 6.6 GB would not survive being
materialized twice in a Python process on a 16 GB machine.

Real weights are used where they happen to be cached, since a real checkpoint
exercises value ranges a `standard_normal * 0.02` fill never reaches.

Run:  uv run python -m mlx_spike.sm5_all_models [--model NAME]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from .common import CACHE_ROOT, REPO_ROOT, RESOURCES, compare
from .onnx_mlx import OnnxMlxGraph, UnsupportedOp

# Where a real checkpoint already lives, keyed by graph stem.
REAL_WEIGHTS = {
    "graph_ext_mitra_classification": CACHE_ROOT / "autogluon__mitra-classifier@main" / "model.safetensors",
    "graph_ext_mitra_regression": CACHE_ROOT / "autogluon__mitra-regressor@main" / "model.safetensors",
}
TENSOR_MAP = {
    "graph_ext_mitra_classification": RESOURCES / "tensor_map_mitra_classification.json",
    "graph_ext_mitra_regression": RESOURCES / "tensor_map_mitra_regression.json",
}


def _mark(key: str, value: object) -> None:
    print(f"SM5_{key}={value}", flush=True)


def _load_equivalence():
    path = REPO_ROOT / "tools" / "gpu_test" / "equivalence.py"
    spec = importlib.util.spec_from_file_location("equivalence", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["equivalence"] = mod
    spec.loader.exec_module(mod)
    return mod


def _signature(graph_path: Path) -> list[str]:
    import onnx

    m = onnx.load(str(graph_path), load_external_data=False)
    return [i.name for i in m.graph.input]


def _feeds(inputs: list[str], t: int, h: int, s: int, seed: int = 1234):
    """The engine's tensor contract. Models split into two families: one takes
    train_size + d as scalars, the other reads the split from y's -100 sentinel
    alone. The signature says which, so it is detected rather than assumed."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, t, h)).astype(np.float32)
    y = np.full((1, t), -100.0, dtype=np.float32)
    y[0, :s] = rng.integers(0, 3, s).astype(np.float32)
    feeds = {"x": x, "y": y}
    if "train_size" in inputs:
        feeds["train_size"] = np.array([s], dtype=np.int64)
    if "d" in inputs:
        feeds["d"] = np.array([h], dtype=np.int64)
    if "cat_mask" in inputs:
        feeds["cat_mask"] = np.zeros((1, h), dtype=np.bool_)
    return feeds, s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="substring filter on the graph stem")
    ap.add_argument("--shapes", default="70x3x60,128x8x100")
    ap.add_argument("--synth-scale", type=float, default=10.0,
                    help="multiplier on synthesized float weights; 1.0 keeps equivalence.py's 0.02 sigma")
    ap.add_argument("--skip-larger-than-gb", type=float, default=2.0,
                    help="skip graphs whose initializers exceed this, to avoid thrashing")
    args = ap.parse_args()

    print("=== S-M5: the ONNX-over-MLX interpreter on every shipped graph ===", flush=True)
    eq = _load_equivalence()
    import onnxruntime as ort

    graphs = sorted(RESOURCES.glob("graph_ext_*.onnx"))
    if args.model:
        graphs = [g for g in graphs if args.model in g.stem]
    # Smallest first. Synthesized initializers are the graph's DECLARED size, so
    # tabfm-v1 fabricates 6.6 GB of random floats, writes them, and has ORT and
    # MLX each read them back -- enough to thrash a 16 GB machine before any of
    # the cheap graphs have been checked. Ordering by declared weight bytes gets
    # twelve verdicts before the expensive one is attempted; --skip-larger-than
    # excludes it outright.
    def declared_bytes(path: Path) -> int:
        import onnx
        m = onnx.load(str(path), load_external_data=False)
        total = 0
        for init in m.graph.initializer:
            if init.data_location == onnx.TensorProto.EXTERNAL:
                meta = {e.key: e.value for e in init.external_data}
                total += int(meta.get("length", 0))
        return total

    sized = sorted(((declared_bytes(g), g) for g in graphs), key=lambda kv: kv[0])
    limit = int(args.skip_larger_than_gb * 2**30)
    graphs = []
    for nbytes, g in sized:
        if nbytes > limit:
            _mark(f"{g.stem}_VERDICT", f"SKIPPED ({nbytes / 2**30:.1f} GB of initializers "
                                       f"> --skip-larger-than-gb {args.skip_larger_than_gb})")
            continue
        graphs.append(g)
    shapes = [tuple(int(v) for v in c.split("x")) for c in args.shapes.split(",")]

    passed, failed, errored, degenerate_cases, inconclusive = [], [], [], [], []
    for graph_path in graphs:
        stem = graph_path.stem
        weights = REAL_WEIGHTS.get(stem)
        tmap = TENSOR_MAP.get(stem)
        use_real = weights is not None and weights.exists()
        try:
            inputs = _signature(graph_path)
            # Materialize the initializers ONCE and give both backends the same
            # file, so a difference can only come from execution.
            arrays = eq.initializer_arrays(graph_path, 0, weights if use_real else None,
                                           tmap if use_real else None)
            if not use_real and args.synth_scale != 1.0:
                # equivalence.py fills synthesized weights at standard_normal *
                # 0.02, which is near-degenerate through a deep stack -- for
                # several of these graphs it drives the logits to a CONSTANT,
                # and then any comparison passes. Rescaling gives the network
                # something to do. Integer initializers are left alone: they
                # are shapes and indices, and scaling them is nonsense.
                arrays = {k: (v * args.synth_scale).astype(v.dtype)
                          if v.dtype.kind == "f" else v
                          for k, v in arrays.items()}
            with tempfile.TemporaryDirectory() as tmp:
                # materialize_external_data returns the DIRECTORY it staged
                # into (it is written to be handed to the plugin as
                # weights_dir), so the graph itself is the copy inside it.
                staged_dir = eq.materialize_external_data(graph_path, arrays, Path(tmp))
                staged = staged_dir / graph_path.name
                sess = ort.InferenceSession(str(staged), providers=["CPUExecutionProvider"])
                interp = OnnxMlxGraph(staged)

                worst = None
                for (t, h, s) in shapes:
                    feeds, train_size = _feeds(inputs, t, h, s)
                    ref = sess.run(["logits"], feeds)[0]
                    t0 = time.perf_counter()
                    got = interp.run(feeds)["logits"]
                    dt = time.perf_counter() - t0
                    c = compare(ref[:, train_size:, :], got[:, train_size:, :])
                    worst = c if worst is None or c.prob_max_abs > worst.prob_max_abs else worst
                    # A reference whose outputs are constant is matched by any
                    # equally-constant candidate, so an exact result there means
                    # nothing. Synthesized weights (standard_normal * 0.02) make
                    # this a live risk: the tolerance probe already found they
                    # collapse mitra's logits into [-0.036, 0.040].
                    q = ref[:, train_size:, :]
                    spread = float(np.ptp(q))
                    degenerate = spread < 1e-9
                    _mark(f"{stem}_{t}x{h}x{s}",
                          f"{'real' if use_real else 'synth'} {c.describe()} "
                          f"ref_spread={spread:.3e}{' DEGENERATE' if degenerate else ''} ms={dt * 1000:.0f}")
                    if degenerate:
                        degenerate_cases.append(f"{stem}@{t}x{h}x{s}")
                # A degenerate reference cannot pass: an exact match against a
                # constant is what a completely broken interpreter also
                # produces. Counting it green would be the single most
                # misleading thing this harness could do.
                if any(c.startswith(stem + "@") for c in degenerate_cases):
                    inconclusive.append(stem)
                    _mark(f"{stem}_VERDICT", "INCONCLUSIVE (reference output is constant)")
                else:
                    ok = worst.argmax_agreement == 1.0 and worst.prob_max_abs <= 1e-4
                    (passed if ok else failed).append(stem)
                    _mark(f"{stem}_VERDICT", "PASS" if ok else "FAIL")
        except UnsupportedOp as exc:
            errored.append((stem, f"unsupported op: {exc}"))
            _mark(f"{stem}_VERDICT", f"UNSUPPORTED ({exc})")
        except Exception as exc:  # noqa: BLE001 - a broken op must name itself
            errored.append((stem, repr(exc)))
            _mark(f"{stem}_VERDICT", f"ERROR {type(exc).__name__}: {str(exc)[:160]}")
            if "-v" in sys.argv:
                traceback.print_exc()

    print()
    _mark("PASSED", f"{len(passed)}/{len(graphs)}")
    if failed:
        _mark("FAILED", ",".join(failed))
    if errored:
        _mark("ERRORED", ",".join(n for n, _ in errored))
    if degenerate_cases:
        # Not a failure of the interpreter -- a failure of the COMPARISON to
        # mean anything. Reported so an exact match on a flat reference is
        # never mistaken for evidence.
        _mark("DEGENERATE_REFERENCE", ",".join(degenerate_cases))
    if inconclusive:
        _mark("INCONCLUSIVE", ",".join(inconclusive))
    ok = not failed and not errored and not inconclusive
    _mark("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
