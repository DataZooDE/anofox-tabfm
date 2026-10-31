"""CI fixture for `tabfm-v1-1` (Google TabFM 1.1.0) — random init, tiny dims.

TabFM 1.1.0 is the SAME architecture as v1 (vendor/tabfm model.py, unchanged)
with three checkpoint-level differences, and the fixture reproduces the two that
can break the engine:

  * an EXPLICIT `decoder_hidden` in config.json (2048) where v1 ships null and
    TabFM falls back to icl_dim * 2. Here it is 24 while the fallback would be
    32 (icl_dim = embed_dim * row_num_cls = 16), so a graph exported WITHOUT it
    would not accept these weights: the decoder/y-encoder shapes differ.
  * BF16 safetensors. v1 publishes F32; 1.1 publishes every tensor as BF16, so
    the engine's BF16 -> F32 upcast is on the load path for the first time for
    a TabFM-family model. The fixture weights are rounded to bf16 first, so the
    stored bytes and the PyTorch reference see exactly the same values.
  * (not reproduced: the smaller embed_dim, 128 vs 256 — any tiny dims are.)

Golden values come from the UNMODIFIED upstream estimator
(vendor/tabfm classifier_and_regressor.py, TabFMClassifier / TabFMRegressor
with n_estimators=1, norm "none", no shuffle, no class shift, no AMP — the
single-member configuration the engine implements) on a table whose cells are
exact in binary floating point, so the SQL test can rebuild it bit-for-bit.
Every row is pinned, context rows included: those come from a forward over the
very tensors upstream's EnsembleGenerator prepares (upstream's predict slices
them away), and the query rows of that forward are asserted to equal upstream's
own predict_proba / predict before anything is written.

Zero Google bytes: the architecture is Apache-2.0, the weights are seeded noise.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from export_onnx import configs as x_configs
from export_onnx import export as x_export
from export_onnx.tabfm_model_patched import TabFM
from make_fixture import upstream

SEED_WEIGHTS = 1337
WEIGHT_SCALE = 1.0  # small scales give a near-constant head on this table (measured); asserted below
N_ROWS, N_CTX, N_FEAT = 20, 12, 5
MIN_MARGIN = 2e-3  # top-1 minus top-2 probability, every row
SOFTMAX_TEMPERATURE = 0.9  # upstream default == engine default

CFG = dict(
    embed_dim=8, max_classes=3, col_num_blocks=2, col_nhead=2, col_num_inds=4,
    row_num_blocks=2, row_nhead=2, row_num_cls=2, icl_num_blocks=2,
    icl_nhead=2, ff_factor=2, feature_group_size=3, num_freq=32,
    decoder_hidden=24,
)
DEFAULT_DECODER_HIDDEN = CFG["embed_dim"] * CFG["row_num_cls"] * 2  # 32


def seeded_model(task: str, seed: int = SEED_WEIGHTS) -> TabFM:
  model = TabFM(**CFG, is_classifier=(task == "classification"))
  gen = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    sd = model.state_dict()
    for key in sorted(sd):
      p = sd[key]
      w = torch.randn(p.shape, generator=gen, dtype=torch.float32) * WEIGHT_SCALE
      p.copy_(w.to(torch.bfloat16).to(torch.float32))  # bf16-exact values
  return model.eval()


def sql_table():
  """The fixture table, from formulas the SQL test repeats verbatim."""
  i = np.arange(N_ROWS)
  cols = {f"f{j + 1}": ((i * 7 + j * 3 + (i * j) % 5) % 11) * 0.25 - 1.0
          for j in range(N_FEAT)}
  X = pd.DataFrame(cols).astype("float64")
  labels = np.array([f"c{(k * 5 + k // 3) % 3}" for k in range(N_CTX)], dtype=object)
  target = np.array([((k * 37) % 17) * 0.25 - 2.0 for k in range(N_CTX)], dtype=np.float64)
  return X, labels, target


def _estimator(car, task, model):
  kw = dict(model=model, n_estimators=1, norm_methods=["none"],
            feat_shuffle_method="none", use_amp=False)
  if task == "classification":
    return car.TabFMClassifier(class_shift=False, **kw)
  return car.TabFMRegressor(**kw)


def _all_rows_forward(est, model, X_test):
  """Forward over the tensors upstream's EnsembleGenerator builds; ALL rows."""
  data = est.ensemble_generator_.transform(est.X_encoder_.transform(X_test))
  Xs, ys, cat_masks, ds, _ = est.ensemble_generator_.prepare_ensemble_tensors(data)
  assert Xs.shape[0] == 1
  T = Xs.shape[1]
  y = np.full((1, T), -100.0, dtype=np.float32)
  y[0, : len(ys[0])] = ys[0]
  with torch.no_grad():
    out = model(torch.as_tensor(Xs, dtype=torch.float32), torch.as_tensor(y),
                torch.tensor([len(ys[0])], dtype=torch.int64),
                cat_mask=torch.as_tensor(np.asarray(cat_masks, dtype=bool)),
                d=torch.as_tensor(np.asarray(ds, dtype=np.int64)))
  return out[0].numpy().astype(np.float64)


def _softmax(z, temperature):
  z = z / temperature
  z = z - z.max(axis=-1, keepdims=True)
  e = np.exp(z)
  return e / e.sum(axis=-1, keepdims=True)


def _classification_ok(car, seed, X, labels) -> bool:
  """Deterministic seed search: the first seed whose classifier gives no
  near-tie row and more than one label in BOTH the context and the query half
  (a head collapsed to one constant would otherwise pass a count check)."""
  model = seeded_model("classification", seed)
  est = _estimator(car, "classification", model)
  est.fit(X.iloc[:N_CTX], labels)
  logits = _all_rows_forward(est, model, X.iloc[N_CTX:])
  proba = _softmax(logits[:, :est.n_classes_], SOFTMAX_TEMPERATURE)
  top2 = np.sort(proba, axis=-1)[:, -2:]
  yhat = proba.argmax(axis=-1)
  return (float((top2[:, 1] - top2[:, 0]).min()) > MIN_MARGIN
          and len(set(yhat[:N_CTX])) > 1 and len(set(yhat[N_CTX:])) > 1)


def build(out: pathlib.Path) -> dict:
  out.mkdir(parents=True, exist_ok=True)
  car = upstream.load()
  X, labels, target = sql_table()
  seed = next(s for s in range(SEED_WEIGHTS, SEED_WEIGHTS + 200)
              if _classification_ok(car, s, X, labels))
  golden = {"_doc": __doc__.strip().splitlines()[0], "rows": N_ROWS, "context_rows": N_CTX,
            "seed": seed}
  hashes = {}
  for task in ("classification", "regression"):
    model = seeded_model(task, seed)
    dec = model.icl_predictor.decoder.layers[0].weight.shape
    assert dec[0] == CFG["decoder_hidden"] != DEFAULT_DECODER_HIDDEN, dec

    st = out / f"model_{task}.safetensors"
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()},
              str(st), metadata={"origin": "anofox-tabfm tabfm-v1-1 CI fixture, random init "
                                           "(bf16), Apache-2.0 architecture, not Google weights"})

    cfg = x_configs.fixture()
    graph = out / f"graph_tabfm_v1_1_{task}.onnx"
    x_export.export_graph(model, graph, dim_rows=cfg.dim_rows,
                          dim_features=cfg.dim_features, example=cfg.example)
    tmap = x_export.postprocess(graph, dict(model.state_dict()))
    x_export.write_tensor_map(out / f"tensor_map_tabfm_v1_1_{task}.json", tmap,
                              task=task, safetensors_rel=f"model_{task}.safetensors")
    parity = x_export.check_parity(graph, model, cfg.parity_shapes)
    assert parity["ok"], parity
    x_export.delete_weight_data(graph)
    x_export.assert_weight_free(graph, tmap)

    est = _estimator(car, task, model)
    if task == "classification":
      est.fit(X.iloc[:N_CTX], labels)
      logits = _all_rows_forward(est, model, X.iloc[N_CTX:])  # [T, max_classes]
      k = est.n_classes_
      proba = _softmax(logits[:, :k], SOFTMAX_TEMPERATURE)
      up = np.asarray(est.predict_proba(X.iloc[N_CTX:]))
      assert np.abs(proba[N_CTX:] - up).max() < 1e-5, np.abs(proba[N_CTX:] - up).max()
      top2 = np.sort(proba, axis=-1)[:, -2:]
      margin = float((top2[:, 1] - top2[:, 0]).min())
      assert margin > MIN_MARGIN, f"near-tie row (margin {margin}); raise WEIGHT_SCALE"
      classes = [str(c) for c in est.classes_]
      yhat = [classes[j] for j in proba.argmax(axis=-1)]
      # not one constant label, in either half (a collapsed head would pass a count check)
      assert len(set(yhat[:N_CTX])) > 1 and len(set(yhat[N_CTX:])) > 1, yhat
      golden[task] = {"classes": classes, "yhat": yhat, "proba": proba.round(6).tolist(),
                      "min_margin": margin, "upstream_max_abs_diff": float(np.abs(proba[N_CTX:] - up).max())}
    else:
      est.fit(X.iloc[:N_CTX], target)
      raw = _all_rows_forward(est, model, X.iloc[N_CTX:])[:, 0]
      ys = est.y_scaler_ if hasattr(est, "y_scaler_") else None
      up = np.asarray(est.predict(X.iloc[N_CTX:]), dtype=np.float64)
      # recover upstream's affine target decode from its own predictions
      mean, scale = float(ys.mean_[0]), float(ys.scale_[0])
      yhat = raw * scale + mean
      assert np.abs(yhat[N_CTX:] - up).max() < 1e-5, np.abs(yhat[N_CTX:] - up).max()
      assert np.unique(yhat.round(4)).size == N_ROWS
      golden[task] = {"yhat": yhat.round(6).tolist(),
                      "upstream_max_abs_diff": float(np.abs(yhat[N_CTX:] - up).max())}
    for p in (st, graph, out / f"tensor_map_tabfm_v1_1_{task}.json"):
      hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
  (out / "golden.json").write_text(json.dumps(golden, indent=1) + "\n")
  hashes["golden.json"] = hashlib.sha256((out / "golden.json").read_bytes()).hexdigest()
  (out / "FIXTURE_SHA256").write_text("".join(f"{h}  {n}\n" for n, h in sorted(hashes.items())))
  return hashes


def main(argv=None) -> int:
  import argparse
  ap = argparse.ArgumentParser(prog="make_fixture_v1_1")
  ap.add_argument("--out", required=True, help="test/fixtures/tabfm_v1_1")
  a = ap.parse_args(argv)
  for name, sha in sorted(build(pathlib.Path(a.out)).items()):
    print(f"{sha}  {name}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
