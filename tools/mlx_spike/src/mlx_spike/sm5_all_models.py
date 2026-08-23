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
    "graph_ext_regression": CACHE_ROOT / "google__tabfm-1.0.0-pytorch@main" / "regression" / "model.safetensors",
    "graph_ext_tabicl_classification": CACHE_ROOT / "jingang__TabICL@main" / "classification" / "model.safetensors",
    "graph_ext_tabicl_regression": CACHE_ROOT / "jingang__TabICL@main" / "regression" / "model.safetensors",
    "graph_ext_orion_bix_classification": CACHE_ROOT / "Lexsi__Orion-BiX@main" / "classification" / "model.safetensors",
    "graph_ext_tabpfn_classification": CACHE_ROOT / "Prior-Labs__TabPFN-v2-clf@main" / "classification" / "model.safetensors",
    "graph_ext_tabpfn_regression": CACHE_ROOT / "Prior-Labs__TabPFN-v2-reg@main" / "regression" / "model.safetensors",
    "graph_ext_tabpfn25_classification": CACHE_ROOT / "Prior-Labs__tabpfn_2_5@main" / "classification" / "model.safetensors",
    "graph_ext_tabpfn25_regression": CACHE_ROOT / "Prior-Labs__tabpfn_2_5@main" / "regression" / "model.safetensors",
    "graph_ext_tabpfn3_classification": CACHE_ROOT / "Prior-Labs__tabpfn_3@main" / "classification" / "model.safetensors",
    # tabpfn-v3 regression has no converted weights: the released checkpoint
    # carries no FullSupportBarDistribution criterion.borders, so
    # tools/export_tabpfn/convert_weights.py cannot build its point-estimate
    # head. Falls back to synthesized weights below.
}
TENSOR_MAP = {
    stem: RESOURCES / f"tensor_map_{stem.replace('graph_ext_', '')}.json"
    for stem in REAL_WEIGHTS
}
# tabfm-v1 keeps its pre-multi-model unqualified map names.
TENSOR_MAP["graph_ext_regression"] = RESOURCES / "tensor_map_regression.json"
TENSOR_MAP = {k: v for k, v in TENSOR_MAP.items() if v.exists()}


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
    """The engine's tensor contract. Two families, and the difference is in
    Y'S LENGTH, not in a sentinel:

      train_size-scalar (tabfm-v1, mitra): y is [1, T], one slot per row, with
        train_size saying where the context ends.
      single_eval_pos (TabPFN, TabICL, Orion): y is [1, TRAIN] -- the graph
        declares a separate dim for it (`y [1, s94]` against `x [1, s27, s53]`)
        and infers the split from that length alone.

    Feeding the second family a full-length y makes it read every row as
    context, and the model then returns a CONSTANT -- which looks like a
    backend bug and is really a malformed input. Detected from the signature
    rather than assumed."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, t, h)).astype(np.float32)
    if "train_size" in inputs:
        y = np.full((1, t), -100.0, dtype=np.float32)
        y[0, :s] = rng.integers(0, 3, s).astype(np.float32)
    else:
        y = rng.integers(0, 3, s).astype(np.float32)[None, :]
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
        w = REAL_WEIGHTS.get(g.stem)
        if w is not None and (w.parent / g.name).exists():
            graphs.append(g)   # in-situ: no staging, so size is irrelevant
            continue
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
            # Fast path: when the real weights already sit next to a copy of
            # this graph -- which is exactly how the engine lays out the cache
            # -- nothing needs staging. That is what makes tabfm-v1 testable at
            # all: materializing its 6.6 GB into a temp dir, on top of the copy
            # ORT and MLX each make, does not fit on a 16 GB machine.
            insitu = weights.parent / graph_path.name if use_real else None
            if insitu is not None and insitu.exists():
                _mark(f"{stem}_INSITU", insitu)
                sess = ort.InferenceSession(str(insitu), providers=["CPUExecutionProvider"])
                interp = OnnxMlxGraph(insitu, weights_path=weights)
                worst, ref_range = None, 0.0
                for (t, h, sp) in shapes:
                    feeds, train_size = _feeds(inputs, t, h, sp)
                    ref = sess.run(["logits"], feeds)[0]
                    t0 = time.perf_counter()
                    got = interp.run(feeds)["logits"]
                    dt = time.perf_counter() - t0
                    c = compare(ref[:, train_size:, :], got[:, train_size:, :])
                    worst = c if worst is None or c.prob_max_abs > worst.prob_max_abs else worst
                    spread = float(np.ptp(ref[:, train_size:, :]))
                    ref_range = max(ref_range, spread)
                    if spread < 1e-9:
                        degenerate_cases.append(f"{stem}@{t}x{h}x{sp}")
                    _mark(f"{stem}_{t}x{h}x{sp}",
                          f"real-insitu {c.describe()} ref_spread={spread:.3e} ms={dt * 1000:.0f}")
                if any(c.startswith(stem + "@") for c in degenerate_cases):
                    inconclusive.append(stem)
                    _mark(f"{stem}_VERDICT", "INCONCLUSIVE (reference output is constant)")
                else:
                    scale = max(ref_range, 1.0)
                    ok = (worst.max_abs <= 1e-4 * scale) if "regression" in stem \
                        else (worst.argmax_agreement == 1.0 and worst.prob_max_abs <= 1e-4)
                    (passed if ok else failed).append(stem)
                    _mark(f"{stem}_VERDICT", "PASS" if ok else "FAIL")
                continue

            with tempfile.TemporaryDirectory() as tmp:
                # materialize_external_data returns the DIRECTORY it staged
                # into (it is written to be handed to the plugin as
                # weights_dir), so the graph itself is the copy inside it.
                staged_dir = eq.materialize_external_data(graph_path, arrays, Path(tmp))
                staged = staged_dir / graph_path.name
                sess = ort.InferenceSession(str(staged), providers=["CPUExecutionProvider"])
                interp = OnnxMlxGraph(staged)

                worst, ref_range = None, 0.0
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
                    ref_range = max(ref_range, spread)
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
                    # Classification outputs are compared after softmax and are
                    # bounded in [0, 1], so 1e-4 absolute is meaningful. A
                    # REGRESSION output is unbounded -- tabicl's spans ~15 --
                    # and holding it to the same absolute bound fails a backend
                    # for fp32 dust. Scale the bound to the reference's range.
                    if "regression" in stem:
                        scale = max(ref_range, 1.0)
                        ok = worst.max_abs <= 1e-4 * scale
                        _mark(f"{stem}_BAR", f"regression: logit_max_abs <= 1e-4 x range({scale:.3g})")
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
