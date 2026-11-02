"""The context-cache pair for TabICL v2: prepare once per support set, then query rows only.

DataZooDE/anofox-tabfm#37, design in docs/CONTEXT_CACHE_PLAN.md ("kv" contract). This is the
supported form of what ``context_spike.py`` measured; ``export_tabicl --split-context`` writes it.

    prepare(x [1,S,H] f32, y [1,S] f32)
        -> hidden_0..hidden_{K-1}              column-ISAB state, one per block (split.py)
        -> icl_k_0, icl_v_0, ..., icl_v_{L-1}  each ICL layer's support keys/values
        -> y_mean, y_std [1,1]                 regression only: the target scaler
        -> context_logits [1,S,C]              the support rows' own outputs, exactly as the
                                               combined graph computes them (C = 1 for regression)
    query(x [1,Q,H] f32, <state by name>)
        -> logits [1,Q,C]                      no y, no support rows

**Why this is exact.** In the combined pass a support row attends only to support rows (the ICL keys
are sliced to ``train_size``), and each query row attends to the support rows only. So the support
rows' outputs are a function of the support set alone, and every query row is independent of the
other query rows. Prepare runs the support rows through the ICL layers and keeps each layer's K/V;
query runs only the query rows through TabICL's own ``cached_kv`` path, which computes the scalable
softmax with the right key count (``src_len`` comes from the cached K).

**Regression** follows the shipped combined graph (``tabicl_patches.ExportWrapper``): the raw targets
are z-scored in-graph (population std, ``std == 0 -> 1``), the model sees ``y_z`` in both the column
embedding and the ICL label embedding, the point estimate is the mean over the quantile head, and the
inverse scaler maps it back to the raw target space. The scaler is part of the cached state, so a
query batch returns the same ``[1,Q,1]`` raw-space value the combined graph would. The pair from
``split.py`` fed raw ``y`` and returned 999 raw quantiles, which is what this replaces.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from tabicl._model.kv_cache import KVCacheEntry

from export_tabicl import configs, export, split

CONTEXT_LOGITS = "context_logits"
SCALER = ("y_mean", "y_std")


def _scaler(y):
    """StandardScaler(ddof=0) on the raw train targets, op for op as ExportWrapper.forward."""
    mean_y = y.mean(dim=1, keepdim=True)
    var_y = ((y - mean_y) ** 2).mean(dim=1, keepdim=True)
    std_y = torch.sqrt(var_y)
    std_y = torch.where(std_y > 0.0, std_y, torch.ones_like(std_y))
    return mean_y, std_y


def _icl_head(m, src):
    icl = m.icl_predictor
    if icl.norm_first:
        src = icl.ln(src)
    return icl.decoder(src)


def _label_embedding(m, y):
    icl = m.icl_predictor
    return icl.y_encoder(y.float()) if icl.max_classes > 0 else icl.y_encoder(y.unsqueeze(-1))


def _point(quantiles, mean_y, std_y):
    """Mean over the quantile head, inverse-scaled: the combined regression graph's output."""
    return quantiles.mean(dim=-1, keepdim=True) * std_y.unsqueeze(-1) + mean_y.unsqueeze(-1)


class PrepareKV(torch.nn.Module):
    """Everything the query rows read, down to each ICL layer's support K/V, plus the support rows'
    own outputs."""

    def __init__(self, model):
        super().__init__()
        # The model is the ONLY submodule, so every initializer is named "m.<checkpoint key>" and the
        # tensor map is built by name. Holding a split.PrepareWrapper as a second submodule names
        # them "prep.m.<key>", which the map then matches by value signature instead -- and random-init
        # biases are all zeros, so it maps ISAB biases to row-interactor biases (see finish()).
        self.m = model
        self.regression = model.max_classes == 0

    def forward(self, x, y):
        if not self.regression:
            *state, head = self._encode(x, y)
            return (*state, head)
        # Regression: the target scaler is computed on EVERY regression export, from the raw targets,
        # exactly as the combined graph does, and travels with the state so query can invert it.
        mean_y, std_y = _scaler(y)
        *state, head = self._encode(x, (y - mean_y) / std_y)
        return (*state, mean_y, std_y, _point(head, mean_y, std_y))

    def _encode(self, x, y):
        """Column-ISAB state, per-layer ICL K/V and the support rows' head output, for labels ``y``
        as the model sees them (z-scored for regression)."""
        *hiddens, support_reps = split.prepare_columns(self.m, x, y)
        icl = self.m.icl_predictor
        out = support_reps + _label_embedding(self.m, y)
        kvs = []
        for block in icl.tf_icl.blocks:
            # Every row here is a support row, so k = v = q is exactly the combined pass's
            # q[..., :train_size, :].
            out, k, v = block(q=out, rope=icl.tf_icl.rope, need_kv=True)
            kvs += [k, v]
        return (*hiddens, *kvs, _icl_head(self.m, out))


class QueryKV(torch.nn.Module):
    """Query rows only, end to end. No labels, no support rows."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.regression = model.max_classes == 0

    def forward(self, x, hiddens, kvs, y_mean=None, y_std=None):
        # `hiddens` and `kvs` are tuples rather than varargs: torch.export flattens *args
        # into one entry and then rejects a dynamic_shapes spec written per tensor.
        ce = self.m.col_embedder
        features = split._grouped_features(ce, x)
        out = ce.in_linear(features)  # no target-aware term: a query batch carries no labels
        for block, hidden in zip(ce.tf_col.blocks, hiddens):
            # Straight to multihead_attn2: the ISAB skip test would re-derive, per batch, a
            # decision the combined pass made with the labels present (split.py).
            out = block.multihead_attn2(out, hidden, hidden)
        emb = split._affine(ce, features, out).transpose(1, 2)
        r = self.m.row_interactor(emb, d=None)
        icl = self.m.icl_predictor
        for i, block in enumerate(icl.tf_icl.blocks):
            r = block(q=r, rope=icl.tf_icl.rope,
                      cached_kv=KVCacheEntry(key=kvs[2 * i], value=kvs[2 * i + 1]))
        head = _icl_head(self.m, r)
        if not self.regression:
            return head
        if y_mean is None or y_std is None:
            raise ValueError("a regression query needs the prepared y_mean and y_std")
        return _point(head, y_mean, y_std)


def state_names(model) -> tuple[list[str], list[str], list[str]]:
    """(hidden_*, icl_k_*/icl_v_*, scaler) -- the tensors prepare emits and query reads."""
    hid = [f"hidden_{i}" for i in range(split.n_context_tensors(model))]
    kv = [f"{kind}_{i}" for i in range(len(model.icl_predictor.tf_icl.blocks))
          for kind in ("icl_k", "icl_v")]
    return hid, kv, (list(SCALER) if model.max_classes == 0 else [])


def pair_paths(out: pathlib.Path, task: str, stem: str = "tabicl"):
    return (out / f"graph_prepare_{stem}_{task}.onnx", out / f"graph_query_{stem}_{task}.onnx")


def map_path_for(graph: pathlib.Path) -> pathlib.Path:
    """graph_ -> tensor_map_, .onnx -> .json: the rule the engine finds every map by."""
    return graph.with_name(graph.name.replace("graph_", "tensor_map_", 1).replace(".onnx", ".json"))


def _example(model, cfg):
    t, h, s = cfg.example
    q = max(2, t - s)
    xs, xq = torch.randn(1, s, h), torch.randn(1, q, h)
    y = (torch.randint(0, model.max_classes, (1, s)).float() if model.max_classes > 0
         else torch.randn(1, s) * 2.0 + 0.5)
    return xs, xq, y


def export_kv(model, prepare_path: pathlib.Path, query_path: pathlib.Path, cfg,
              opset: int = configs.OPSET) -> None:
    """Export the pair WITH its weights (``.onnx.data``) -- call ``finish`` to make it weight-free.

    Kept in two steps so the parity check can run the graphs as exported, before the weights go.
    """
    prep, qry = PrepareKV(model).train(), QueryKV(model).train()
    xs, xq, y = _example(model, cfg)
    S = torch.export.Dim(cfg.dim_train[0], min=cfg.dim_train[1], max=cfg.dim_train[2])
    Q = torch.export.Dim("query", min=2, max=cfg.dim_train[2])
    H = torch.export.Dim(cfg.dim_features[0], min=cfg.dim_features[1], max=cfg.dim_features[2])
    # G+C, the column-group axis of the cached ISAB state. Declared in its own right: torch.export
    # cannot see that it is a function of H through feature_grouping (split.py).
    G = torch.export.Dim("groups", min=2, max=4 * cfg.dim_features[2])
    hid, kv, scaler = state_names(model)
    prepare_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        prepared = prep(xs, y)
        torch.onnx.export(prep, (xs, y), str(prepare_path), dynamo=True,
                          dynamic_shapes=({1: S, 2: H}, {1: S}), opset_version=opset,
                          input_names=["x", "y"], output_names=hid + kv + scaler + [CONTEXT_LOGITS],
                          external_data=True, optimize=False)
        nk, nkv = len(hid), len(kv)
        args = (xq, tuple(prepared[:nk]), tuple(prepared[nk:nk + nkv]))
        dyn = ({1: Q, 2: H}, tuple({1: G} for _ in hid), tuple({2: S} for _ in kv))
        if scaler:  # regression only: y_mean, y_std as two plain [1,1] arguments
            args += tuple(prepared[nk + nkv:nk + nkv + len(scaler)])
            dyn += (None, None)
        torch.onnx.export(qry, args, str(query_path), dynamo=True, dynamic_shapes=dyn,
                          opset_version=opset, input_names=["x"] + hid + kv + scaler,
                          output_names=["logits"], external_data=True, optimize=False)


def finish(model, path: pathlib.Path, task: str, safetensors_rel: str) -> dict:
    """Same shipping treatment as the combined graph: externalize every checkpoint initializer,
    write the half's own tensor map, drop the weights. Each half gets its own map because neither
    half declares every checkpoint tensor (the query half has no label encoders) and ORT rejects an
    injected initializer its graph does not declare."""
    tmap = export.postprocess(path, dict(model.state_dict()))
    # Every initializer must map BY NAME. A signature (value-hash) match is ambiguous for tensors that
    # are equal at export time -- every zero-initialised bias -- and injects the wrong real weight
    # without any error: the graph runs and is simply wrong.
    by_value = {k: v for k, v in tmap["initializers"].items() if export._norm_name(k) != v}
    if by_value or tmap["transforms"]:
        raise RuntimeError(f"{path.name}: initializers mapped by value, not by name: "
                           f"{sorted(by_value.items())[:5]} transforms={tmap['transforms']}")
    export.write_tensor_map(map_path_for(path), tmap, task=task, safetensors_rel=safetensors_rel)
    export.delete_weight_data(path)
    export.assert_weight_free(path, tmap)
    return tmap


# --------------------------------------------------------------------------------------------------
# Parity
# --------------------------------------------------------------------------------------------------

def _ort_session(graph: pathlib.Path, weights: dict | None = None, threads: int = 4):
    """A graph as ORT sees it in the engine: weight-free, initializers injected through its map.
    ``weights=None`` runs a graph that still has its ``.onnx.data``."""
    import onnx
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    if weights is None:
        return ort.InferenceSession(str(graph), so, providers=["CPUExecutionProvider"])
    tmap = json.loads(map_path_for(graph).read_text())["initializers"]
    proto = onnx.load(str(graph), load_external_data=False)
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


class Pair:
    """The combined graph and the kv pair, side by side, in ORT."""

    def __init__(self, combined, prepare, query):
        self.combined, self.prepare, self.query = combined, prepare, query
        self.state = [i.name for i in query.get_inputs() if i.name != "x"]

    def run_combined(self, xs, xq, y):
        return self.combined.run(["logits"], {"x": np.concatenate([xs, xq], 1), "y": y})[0]

    def run_prepare(self, xs, y):
        names = self.state + [CONTEXT_LOGITS]
        out = dict(zip(names, self.prepare.run(names, {"x": xs, "y": y})))
        return out.pop(CONTEXT_LOGITS), out

    def run_query(self, xq, state):
        return self.query.run(["logits"], {"x": xq, **state})[0]


def _max_abs(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def compare(pair: Pair, xs, xq, y, n_out: int, regression: bool) -> dict:
    """Every number the gate needs for one table. ``n_out`` = real classes (1 for regression)."""
    s = xs.shape[1]
    base = pair.run_combined(xs, xq, y)[0, :, :n_out]
    ctx, state = pair.run_prepare(xs, y)
    test = pair.run_query(xq, state)[0, :, :n_out]
    h = max(1, xq.shape[1] // 3)
    rebatched = np.concatenate([pair.run_query(xq[:, :h], state)[0],
                                pair.run_query(xq[:, h:], state)[0]], 0)[:, :n_out]
    # Control: a context prepared from permuted labels must move the answers. A graph or a model that
    # ignored its context would pass every identity check above and fail this one.
    perm = np.random.default_rng(1).permutation(s)
    _, st_perm = pair.run_prepare(xs, y[:, perm])
    control = pair.run_query(xq, st_perm)[0, :, :n_out]
    r: dict = {"S": s, "Q": int(xq.shape[1]), "H": int(xs.shape[2]), "out": n_out,
         "test_max_abs": _max_abs(base[s:], test),
         "context_max_abs": _max_abs(base[:s], ctx[0, :, :n_out]),
         "rebatched_max_abs": _max_abs(test, rebatched),
         "control_max_abs": _max_abs(test, control)}
    if regression:
        scale = float(np.std(y)) or 1.0
        for k in ("test", "context", "rebatched", "control"):
            r[f"{k}_rel"] = r[f"{k}_max_abs"] / scale
    else:
        r["test_label_diffs"] = int((base[s:].argmax(-1) != test.argmax(-1)).sum())
        r["context_label_diffs"] = int((base[:s].argmax(-1) != ctx[0, :, :n_out].argmax(-1)).sum())
        r["control_label_diffs"] = int((test.argmax(-1) != control.argmax(-1)).sum())
    return r


# Budgets for the real-weight gate. The spike measured <= 1.2e-5 on logits (fp32 reduction order:
# the K/V come from a need_kv projection rather than a slice). Regression is held to the same bound,
# relative to the target's spread, since its output is in raw target units.
LOGIT_TOL = 1e-4
REL_TOL = 1e-4
# The control must move the answers by far more than the tolerance, or the identity checks prove
# nothing about this model (a near-constant random-init fixture moves by ~1e-8).
CONTROL_MIN = 1e-2


def verdict(rows: list[dict], regression: bool) -> dict:
    fails = []
    for r in rows:
        name = f"{r.get('table', '?')} S={r['S']} Q={r['Q']}"
        if regression:
            if r["test_rel"] > REL_TOL or r["context_rel"] > REL_TOL:
                fails.append(f"{name}: |delta| test {r['test_rel']:.2e} context {r['context_rel']:.2e} "
                             f"(relative to std(y); budget {REL_TOL:.0e})")
            if r["control_rel"] < CONTROL_MIN:
                fails.append(f"{name}: permuted-label control moved only {r['control_rel']:.2e}")
        else:
            if r["test_label_diffs"] or r["context_label_diffs"]:
                fails.append(f"{name}: label diffs test {r['test_label_diffs']} context "
                             f"{r['context_label_diffs']}")
            if r["test_max_abs"] > LOGIT_TOL or r["context_max_abs"] > LOGIT_TOL:
                fails.append(f"{name}: |dlogit| test {r['test_max_abs']:.2e} context "
                             f"{r['context_max_abs']:.2e} (budget {LOGIT_TOL:.0e})")
            if r["control_max_abs"] < CONTROL_MIN:
                fails.append(f"{name}: permuted-label control moved only {r['control_max_abs']:.2e}")
        # A query row must not depend on which other query rows share its call: the engine's
        # query_batch chunking relies on it.
        if (r["rebatched_rel"] > REL_TOL) if regression else (r["rebatched_max_abs"] > LOGIT_TOL):
            fails.append(f"{name}: re-batching moved the answers by {r['rebatched_max_abs']:.2e}")
    return {"ok": not fails, "failures": fails, "rows": rows}


def real_tables(task: str):
    """Real and synthetic tables as (name, x [N,H] f32, target [N]). The features go in raw: both
    graphs see identical inputs, and TabICL normalises inside the graph."""
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits

    rng = np.random.default_rng(0)
    out = []
    if task == "classification":
        for name, loader in (("breast_cancer", load_breast_cancer), ("digits", load_digits)):
            d = loader()
            out.append((name, d.data.astype(np.float32), d.target.astype(np.int64)))
        x = rng.standard_normal((1200, 100)).astype(np.float32)
        out.append(("synthetic_random_labels", x, rng.integers(0, 4, 1200)))
    else:
        d = load_diabetes()
        out.append(("diabetes", d.data.astype(np.float32), d.target.astype(np.float64)))
        x = rng.standard_normal((1200, 20)).astype(np.float32)
        t = np.sin(x[:, 0] * 2) * 50 + x[:, 1] ** 2 * 10 + x[:, 2] * x[:, 3] * 5 + 300.0
        out.append(("synthetic_nonlinear", x, t + rng.standard_normal(1200)))
    return out


def gate(prepare: pathlib.Path, query: pathlib.Path, combined: pathlib.Path, weights_path: pathlib.Path,
         task: str, shapes=((64, 16), (256, 128), (375, 128), (1000, 128)), threads: int = 4) -> dict:
    """The pair against the SHIPPED combined graph, on real weights injected through each graph's map.

    Classification: zero label differences and |dlogit| <= LOGIT_TOL on test AND context rows.
    Regression: |dyhat| / std(y) <= REL_TOL. Both: re-batching identical, permuted-label control moves.
    """
    from safetensors.numpy import load_file

    w = load_file(str(weights_path))
    pair = Pair(_ort_session(combined, w, threads), _ort_session(prepare, w, threads),
                _ort_session(query, w, threads))
    regression = task == "regression"
    rng = np.random.default_rng(0)
    rows = []
    for name, x, t in real_tables(task):
        perm = rng.permutation(len(x))
        for s_, q_ in shapes:
            if s_ + q_ > len(x):
                continue
            i_s, i_q = perm[:s_], perm[s_:s_ + q_]
            y = t[None, i_s].astype(np.float32)
            n_out = 1 if regression else int(t.max()) + 1
            r = compare(pair, x[None, i_s], x[None, i_q], y, n_out, regression)
            r["table"] = name
            rows.append(r)
    return verdict(rows, regression)


def check_against_torch(prepare_path: pathlib.Path, query_path: pathlib.Path, model, shapes) -> float:
    """Worst |difference| over ALL rows (context and test) between the chained pair, run in ORT as
    exported (weights still attached), and one PyTorch forward of the combined ExportWrapper.

    ``shapes`` are (S, Q, H) triples and should differ from the export example, so a baked dimension
    shows up as a failure rather than as a pass. This runs on whatever weights ``model`` has (random
    at export time); the real-weight gate is ``gate``.
    """
    from export_tabicl.tabicl_patches import ExportWrapper

    wrapper = ExportWrapper(model).train()
    prep, qry = _ort_session(prepare_path), _ort_session(query_path)
    state = [i.name for i in qry.get_inputs() if i.name != "x"]
    worst = 0.0
    gen = torch.Generator().manual_seed(3)
    for s, q, h in shapes:
        xs, xq = torch.randn(1, s, h, generator=gen), torch.randn(1, q, h, generator=gen)
        y = (torch.randint(0, model.max_classes, (1, s), generator=gen).float() if model.max_classes > 0
             else torch.randn(1, s, generator=gen) * 3.0 - 1.0)
        with torch.no_grad():
            base = wrapper(torch.cat([xs, xq], 1), y).numpy()
        out = prep.run(state + [CONTEXT_LOGITS], {"x": xs.numpy(), "y": y.numpy()})
        st = dict(zip(state, out[:-1]))
        test = qry.run(["logits"], {"x": xq.numpy(), **st})[0]
        got = np.concatenate([out[-1], test], 1)
        worst = max(worst, _max_abs(base, got))
    return worst


def export_pair(model, out: pathlib.Path, task: str, cfg, *, stem: str = "tabicl",
                safetensors_rel: str | None = None, check: bool = True) -> dict:
    """Export, check against PyTorch at shapes other than the example, and ship weight-free."""
    prep, qry = pair_paths(out, task, stem)
    export_kv(model, prep, qry, cfg)
    report: dict = {"prepare": prep.name, "query": qry.name}
    if check:
        worst = check_against_torch(prep, qry, model,
                                    [(s_, max(2, t_ - s_), h_) for t_, h_, s_ in cfg.parity_shapes])
        report["torch_parity_worst"] = worst
        report["torch_parity_ok"] = worst < export.PARITY_TOL
    rel = safetensors_rel or f"{task}/model.safetensors"
    for path in (prep, qry):
        tmap = finish(model, path, task, rel)
        report[f"{path.name}_initializers"] = len(tmap["initializers"])
    return report


def main(argv=None) -> int:
    """uv run gate_context_cache --pair DIR [--combined DIR] [--weights DIR] [--task T ...]

    The ship gate for a context-cache pair: against the SHIPPED combined graph (resources/ by default)
    on the real weights, every task. Exit 1 on any failure. Writes the numbers as JSON to stdout.
    """
    import argparse
    import os

    repo = pathlib.Path(__file__).resolve().parents[4]
    ap = argparse.ArgumentParser(prog="gate_context_cache")
    ap.add_argument("--pair", required=True, help="directory holding graph_{prepare,query}_tabicl_<task>.onnx + maps")
    ap.add_argument("--combined", default=str(repo / "resources"))
    ap.add_argument("--weights", default=os.environ.get(
        "TABICL_WEIGHTS",
        str(pathlib.Path("~/.cache/anofox-tabfm/jingang__TabICL@4dcd344ece2c00be9e831fdd35bed57b5ad83e19")
            .expanduser())),
        help="directory with <task>/model.safetensors (the pinned revision)")
    ap.add_argument("--task", action="append", choices=["classification", "regression"])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    ok = True
    report = {}
    for task in args.task or ["classification", "regression"]:
        prep, qry = pair_paths(pathlib.Path(args.pair), task)
        res = gate(prep, qry, pathlib.Path(args.combined) / f"graph_tabicl_{task}.onnx",
                   pathlib.Path(args.weights) / task / "model.safetensors", task, threads=args.threads)
        report[task] = res
        ok &= res["ok"]
        for f in res["failures"]:
            print(f"[gate_context_cache] {task}: FAIL {f}", flush=True)
    print(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
