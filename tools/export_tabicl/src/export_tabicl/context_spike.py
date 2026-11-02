"""Spike for DataZooDE/anofox-tabfm#37: how much of a tabicl-v2 call can a support-set context save,
and is the saving exact? Results and the design they feed: docs/CONTEXT_CACHE_PLAN.md.

Three contracts are measured against the SHIPPED combined graph (resources/graph_tabicl_*.onnx):

* ``pair`` -- what ``export_tabicl --split-context`` writes today (split.py) and what the engine runs
  under ``SET anofox_tabfm_context_cache = true``: prepare(x_S, y) -> col-ISAB ``hidden_*`` +
  ``support_reps``; query(x_Q, y, state) -> logits. The engine gets the context rows' fitted values
  from a second query pass over the S support rows.
* ``fitted`` -- the same pair, with a prepare graph that ALSO emits the context rows' own logits,
  computed as the combined graph computes them (support reps + label embedding through the ICL stage).
  Fitted values become identical to the combined graph, and a cache miss no longer pays a query pass
  over S.
* ``kv`` -- prepare additionally runs the ICL stage over the support rows and emits each ICL layer's
  key/value projections; query runs the query rows only, attending to the cached K/V (TabICL's own
  ``cached_kv`` path in ``MultiheadAttentionBlock``). Today's query graph re-runs the whole 12-layer
  ICL transformer over all S support rows on every call; this one does not, and takes no ``y``.

All graphs are weight-free and the weights are INJECTED from the safetensors file through each
graph's tensor map (``add_external_initializers``), the way the extension does it.

    uv run spike_context_cache --graphs DIR --out DIR [--weights ~/.cache/anofox-tabfm/jingang__TabICL@main]

``--graphs`` holds the combined graph + map (copy them from resources/) and the pair from
``export_tabicl --config real --split-context``. Writes the two new graph pairs under --out and
``spike_report.json``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from safetensors.numpy import load_file
from tabicl._model.kv_cache import KVCacheEntry

from export_tabicl import configs, export, split
from export_tabicl.tabicl_patches import build_model

REPO = pathlib.Path(__file__).resolve().parents[4]
TASK = "classification"


def _icl_head(m, src):
    icl = m.icl_predictor
    if icl.norm_first:
        src = icl.ln(src)
    return icl.decoder(src)


class PrepareFittedWrapper(torch.nn.Module):
    """split.PrepareWrapper plus the context rows' logits as the combined graph computes them.

    In the combined pass a support row attends only to support rows (the ICL keys are sliced to
    ``train_size``), so its logits are a function of the support set alone.
    """

    def __init__(self, model):
        super().__init__()
        self.prep = split.PrepareWrapper(model)
        self.m = model

    def forward(self, x, y):
        *hiddens, support_reps = self.prep(x, y)
        # _icl_predictions adds the label embedding to R in place; give it its own copy.
        context_logits = self.m.icl_predictor._icl_predictions(support_reps.clone(), y)
        return (*hiddens, support_reps, context_logits)


class PrepareKVWrapper(torch.nn.Module):
    """Everything the query rows read, down to each ICL layer's support K/V."""

    def __init__(self, model):
        super().__init__()
        self.prep = split.PrepareWrapper(model)
        self.m = model

    def forward(self, x, y):
        *hiddens, support_reps = self.prep(x, y)
        icl = self.m.icl_predictor
        out = support_reps + icl.y_encoder(y.float())
        kvs = []
        for block in icl.tf_icl.blocks:
            # Every row here is a support row, so k = v = q is exactly the combined pass's
            # q[..., :train_size, :].
            out, k, v = block(q=out, rope=icl.tf_icl.rope, need_kv=True)
            kvs += [k, v]
        return (*hiddens, *kvs, _icl_head(self.m, out))


class QueryKVWrapper(torch.nn.Module):
    """Query rows only, end to end. No labels, no support rows."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x, hiddens, kvs):
        ce = self.m.col_embedder
        features = split._grouped_features(ce, x)
        out = ce.in_linear(features)
        for block, hidden in zip(ce.tf_col.blocks, hiddens):
            out = block.multihead_attn2(out, hidden, hidden)
        emb = split._affine(ce, features, out).transpose(1, 2)
        r = self.m.row_interactor(emb, d=None)
        icl = self.m.icl_predictor
        for i, block in enumerate(icl.tf_icl.blocks):
            r = block(q=r, rope=icl.tf_icl.rope,
                      cached_kv=KVCacheEntry(key=kvs[2 * i], value=kvs[2 * i + 1]))
        return _icl_head(self.m, r)


def _finish(model, path: pathlib.Path):
    tmap = export.postprocess(path, dict(model.state_dict()))
    export.write_tensor_map(path.with_name(path.name.replace("graph_", "tensor_map_", 1)
                                           .replace(".onnx", ".json")),
                            tmap, task=TASK, safetensors_rel=f"{TASK}/model.safetensors")
    export.delete_weight_data(path)
    export.assert_weight_free(path, tmap)


def export_new_graphs(model, out: pathlib.Path, cfg) -> None:
    """graph_prepare_fitted_*, graph_prepare_kv_*, graph_query_kv_* (+ maps), weight-free."""
    out.mkdir(parents=True, exist_ok=True)
    t, h, s = cfg.example
    q = max(2, t - s)
    xs, xq = torch.randn(1, s, h), torch.randn(1, q, h)
    y = torch.randint(0, model.max_classes, (1, s)).float()
    S = torch.export.Dim(cfg.dim_train[0], min=cfg.dim_train[1], max=cfg.dim_train[2])
    Q = torch.export.Dim("query", min=2, max=cfg.dim_train[2])
    H = torch.export.Dim(cfg.dim_features[0], min=cfg.dim_features[1], max=cfg.dim_features[2])
    G = torch.export.Dim("groups", min=2, max=4 * cfg.dim_features[2])
    nk = split.n_context_tensors(model)
    nl = len(model.icl_predictor.tf_icl.blocks)
    hid = [f"hidden_{i}" for i in range(nk)]
    kv = [f"{kind}_{i}" for i in range(nl) for kind in ("icl_k", "icl_v")]

    pf = PrepareFittedWrapper(model).train()
    pk = PrepareKVWrapper(model).train()
    qk = QueryKVWrapper(model).train()
    with torch.no_grad():
        p = out / f"graph_prepare_fitted_tabicl_{TASK}.onnx"
        torch.onnx.export(pf, (xs, y), str(p), dynamo=True,
                          dynamic_shapes=({1: S, 2: H}, {1: S}), opset_version=configs.OPSET,
                          input_names=["x", "y"],
                          output_names=hid + ["support_reps", "context_logits"],
                          external_data=True, optimize=False)
        _finish(model, p)

        prepared = pk(xs, y)
        p = out / f"graph_prepare_kv_tabicl_{TASK}.onnx"
        torch.onnx.export(pk, (xs, y), str(p), dynamo=True,
                          dynamic_shapes=({1: S, 2: H}, {1: S}), opset_version=configs.OPSET,
                          input_names=["x", "y"], output_names=hid + kv + ["context_logits"],
                          external_data=True, optimize=False)
        _finish(model, p)

        p = out / f"graph_query_kv_tabicl_{TASK}.onnx"
        torch.onnx.export(qk, (xq, tuple(prepared[:nk]), tuple(prepared[nk:nk + 2 * nl])), str(p),
                          dynamo=True,
                          dynamic_shapes=({1: Q, 2: H}, tuple({1: G} for _ in range(nk)),
                                          tuple({2: S} for _ in range(2 * nl))),
                          opset_version=configs.OPSET, input_names=["x"] + hid + kv,
                          output_names=["logits"], external_data=True, optimize=False)
        _finish(model, p)


def session(graph: pathlib.Path, weights: dict, threads: int):
    """A weight-free graph with its initializers injected, the way the engine does it."""
    tmap_path = graph.with_name(graph.name.replace("graph_", "tensor_map_", 1)
                                .replace(".onnx", ".json"))
    tmap = json.loads(tmap_path.read_text())["initializers"]
    proto = onnx.load(str(graph), load_external_data=False)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    names, values, keep = [], [], []
    for init in proto.graph.initializer:
        key = tmap.get(init.name)
        if key is None:
            continue
        arr = np.ascontiguousarray(weights[key].astype(np.float32))
        keep.append(arr)
        names.append(init.name)
        values.append(ort.OrtValue.ortvalue_from_numpy(arr))
    so.add_external_initializers(names, values)
    sess = ort.InferenceSession(proto.SerializeToString(), so, providers=["CPUExecutionProvider"])
    sess._keep = (keep, values)  # ORT holds references only
    return sess


class Graphs:
    def __init__(self, gdir: pathlib.Path, ndir: pathlib.Path, weights: dict, threads: int):
        g = lambda d, half: session(d / f"graph_{half}tabicl_{TASK}.onnx", weights, threads)  # noqa
        self.combined = g(gdir, "")
        self.prepare, self.query = g(gdir, "prepare_"), g(gdir, "query_")
        self.prepare_fitted = g(ndir, "prepare_fitted_")
        self.prepare_kv, self.query_kv = g(ndir, "prepare_kv_"), g(ndir, "query_kv_")

    @staticmethod
    def _run(sess, feed, outputs=None):
        names = outputs or [o.name for o in sess.get_outputs()]
        return dict(zip(names, sess.run(names, feed)))

    @staticmethod
    def _state_for(sess, state):
        want = {i.name for i in sess.get_inputs()}
        return {k: v for k, v in state.items() if k in want}

    def combined_logits(self, xs, xq, y):
        return self.combined.run(["logits"], {"x": np.concatenate([xs, xq], 1), "y": y})[0]

    def run_prepare(self, xs, y):
        return self._run(self.prepare, {"x": xs, "y": y})

    def run_query(self, xq, y, state):
        return self.query.run(["logits"], {"x": xq, "y": y, **self._state_for(self.query, state)})[0]

    def run_prepare_fitted(self, xs, y):
        out = self._run(self.prepare_fitted, {"x": xs, "y": y})
        return out.pop("context_logits"), out

    def run_prepare_kv(self, xs, y):
        out = self._run(self.prepare_kv, {"x": xs, "y": y})
        return out.pop("context_logits"), out

    def run_query_kv(self, xq, state):
        return self.query_kv.run(["logits"], {"x": xq, **self._state_for(self.query_kv, state)})[0]


def _cmp(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.abs(a - b).max()), int((a.argmax(-1) != b.argmax(-1)).sum())


def parity(gr: Graphs, xs, xq, y, n_classes: int) -> dict:
    s, c = xs.shape[1], n_classes
    base = gr.combined_logits(xs, xq, y)[0, :, :c]
    bc, bt = base[:s], base[s:]
    st = gr.run_prepare(xs, y)
    q_pair = gr.run_query(xq, y, st)[0, :, :c]
    ctx_pair = gr.run_query(xs, y, st)[0, :, :c]  # how the engine gets fitted values today
    ctx_fit, st_f = gr.run_prepare_fitted(xs, y)
    q_fit = gr.run_query(xq, y, st_f)[0, :, :c]
    ctx_kv, st_kv = gr.run_prepare_kv(xs, y)
    q_kv = gr.run_query_kv(xq, st_kv)[0, :, :c]
    h = max(1, xq.shape[1] // 3)
    q_kv_rebatched = np.concatenate([gr.run_query_kv(xq[:, :h], st_kv)[0],
                                     gr.run_query_kv(xq[:, h:], st_kv)[0]], 0)[:, :c]
    # Control: a state prepared from permuted labels must move the answers. A model or graph that
    # ignored its context would pass every parity check above and fail this one.
    _, st_perm = gr.run_prepare_kv(xs, y[:, np.random.default_rng(1).permutation(s)])
    q_perm = gr.run_query_kv(xq, st_perm)[0, :, :c]
    r = {"S": s, "Q": xq.shape[1], "H": xs.shape[2], "classes": c}
    for name, (a, b) in {
        "test_pair": (bt, q_pair), "test_fitted": (bt, q_fit), "test_kv": (bt, q_kv),
        "test_kv_rebatched": (q_kv, q_kv_rebatched),
        "ctx_pair": (bc, ctx_pair), "ctx_fitted": (bc, ctx_fit[0, :, :c]), "ctx_kv": (bc, ctx_kv[0, :, :c]),
        "control_permuted_labels": (q_kv, q_perm),
    }.items():
        r[f"{name}_max_abs"], r[f"{name}_label_diffs"] = _cmp(a, b)
    return r


def timings(gr: Graphs, rng, s, q, h, classes, reps) -> dict:
    """Median wall and CPU seconds per graph run. The variants are INTERLEAVED round by round, so
    load from other processes on a shared box lands on all of them alike rather than on whichever
    happened to run during a spike. CPU time is process CPU (all ORT threads; spinning is off)."""
    xs = rng.standard_normal((1, s, h)).astype(np.float32)
    xq = rng.standard_normal((1, q, h)).astype(np.float32)
    y = rng.integers(0, classes, (1, s)).astype(np.float32)
    st = gr.run_prepare(xs, y)
    _, st_kv = gr.run_prepare_kv(xs, y)
    fns = {
        "combined": lambda: gr.combined_logits(xs, xq, y),
        "pair_prepare": lambda: gr.run_prepare(xs, y),
        "pair_query_S": lambda: gr.run_query(xs, y, st),
        "pair_query": lambda: gr.run_query(xq, y, st),
        "fitted_prepare": lambda: gr.run_prepare_fitted(xs, y),
        "kv_prepare": lambda: gr.run_prepare_kv(xs, y),
        "kv_query": lambda: gr.run_query_kv(xq, st_kv),
    }
    for fn in fns.values():
        fn()  # warm-up: the first run plans and allocates
    wall = {k: [] for k in fns}
    cpu = {k: [] for k in fns}
    for _ in range(reps):
        for k, fn in fns.items():
            w0, c0 = time.perf_counter(), time.process_time()
            fn()
            wall[k].append(time.perf_counter() - w0)
            cpu[k].append(time.process_time() - c0)
    r = {"S": s, "Q": q, "H": h}
    for kind, src in (("wall", wall), ("cpu", cpu)):
        m = {k: statistics.median(v) for k, v in src.items()}
        m["pair_cold"] = m["pair_prepare"] + m["pair_query_S"] + m["pair_query"]  # the engine today
        m["fitted_cold"] = m["fitted_prepare"] + m["pair_query"]
        m["kv_cold"] = m["kv_prepare"] + m["kv_query"]
        r[kind] = m
    r["kv_state_bytes"] = int(sum(v.nbytes for v in st_kv.values()))
    r["pair_state_bytes"] = int(sum(v.nbytes for k, v in st.items()
                                    if k.startswith(("hidden_", "support_reps"))))
    return r


def real_tables():
    from sklearn.datasets import load_breast_cancer, load_digits
    out = []
    for name, loader in (("breast_cancer", load_breast_cancer), ("digits", load_digits)):
        d = loader()
        x = d.data.astype(np.float32)
        x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
        out.append((name, x.astype(np.float32), d.target.astype(np.int64)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spike_context_cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--weights", default=str(pathlib.Path.home()
                                            / ".cache/anofox-tabfm/jingang__TabICL@main"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--shapes", default="64x16,64x128,375x16,375x128,1000x16,1000x128")
    ap.add_argument("--features", type=int, default=100)
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out)
    report = {"threads": args.threads, "reps": args.reps, "onnxruntime": ort.__version__}

    # ---- the committed CI fixture: its own pair, its golden inputs ------------------------------
    if args.skip_parity:
        return _timings_only(args, out, report)
    fx = REPO / "test/fixtures/tabicl"
    fcfg = configs.fixture()
    if not args.skip_export:
        export_new_graphs(build_model(TASK, fcfg.model_kwargs, seed=0), out / "fixture", fcfg)
    fg = Graphs(fx, out / "fixture", load_file(str(fx / f"model_{TASK}.safetensors")), args.threads)
    gold = json.loads((fx / f"golden_{TASK}.json").read_text())
    gx = np.asarray(gold["inputs"]["x"], np.float32)
    gy = np.asarray(gold["inputs"]["y"], np.float32)
    s = gold["inputs"]["train_size"]
    report["fixture"] = parity(fg, gx[:, :s], gx[:, s:], gy, 3)
    report["fixture"]["golden_vs_combined_max_abs"] = float(np.abs(
        np.asarray(gold["logits"], np.float32)[0, s:]
        - fg.combined_logits(gx[:, :s], gx[:, s:], gy)[0, s:]).max())

    # ---- real tabicl-classifier-v2 weights ------------------------------------------------------
    rcfg = configs.real(TASK)
    if not args.skip_export:
        export_new_graphs(build_model(TASK, rcfg.model_kwargs, seed=0), out / "real", rcfg)
    rg = Graphs(pathlib.Path(args.graphs), out / "real",
                load_file(str(pathlib.Path(args.weights) / TASK / "model.safetensors")), args.threads)

    rng = np.random.default_rng(0)
    report["real"] = []
    for name, x, t in real_tables():
        n_classes = int(t.max()) + 1
        perm = rng.permutation(len(x))
        for s_, q_ in ((64, 16), (375, 128)):
            i_s, i_q = perm[:s_], perm[s_:s_ + q_]
            ys = t[None, i_s].astype(np.float32)
            p = parity(rg, x[None, i_s], x[None, i_q], ys, n_classes)
            base = rg.combined_logits(x[None, i_s], x[None, i_q], ys)[0, s_:, :n_classes]
            p["table"], p["test_accuracy"] = name, float((base.argmax(1) == t[i_q]).mean())
            report["real"].append(p)
    for s_, q_ in ((64, 16), (1000, 128)):
        xs = rng.standard_normal((1, s_, args.features)).astype(np.float32)
        xq = rng.standard_normal((1, q_, args.features)).astype(np.float32)
        p = parity(rg, xs, xq, rng.integers(0, 4, (1, s_)).astype(np.float32), 4)
        p["table"] = "synthetic_random_labels"
        report["real"].append(p)
    print(json.dumps({k: report[k] for k in ("fixture", "real")}, indent=1), flush=True)

    return _timings_only(args, out, report, rg)


def _timings_only(args, out, report, rg=None) -> int:
    if rg is None:
        rg = Graphs(pathlib.Path(args.graphs), out / "real",
                    load_file(str(pathlib.Path(args.weights) / TASK / "model.safetensors")),
                    args.threads)
    rng = np.random.default_rng(7)
    report["timings"] = []
    for shp in args.shapes.split(","):
        s_, q_ = map(int, shp.split("x"))
        r = timings(rg, rng, s_, q_, args.features, 4, args.reps)
        report["timings"].append(r)
        print(json.dumps(r), flush=True)

    name = "spike_timings.json" if args.skip_parity else "spike_report.json"
    (out / name).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
