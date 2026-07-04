"""Golden preprocess/ensemble vectors: structural + semantic asserts.

These encode the upstream semantics the C++ port must reproduce:
first-appearance ordinal encoding with min_frequency=2 and -1 unknowns,
alphabetical label encoding, mean imputation, datetime expansion +
train-mean NaT fill, unique filter, z-score + outlier clipping, ensemble
member draws for seed 42, NNLS blend + softmax temperature 0.9.
"""

import json
import math

import numpy as np
import pytest

from make_fixture import golden_preprocess


@pytest.fixture(scope="session")
def payload(tmp_path_factory):
  out = tmp_path_factory.mktemp("golden") / "golden_preprocess.json"
  p = golden_preprocess.generate(out)
  # must round-trip through strict JSON (no NaN/Inf leaks)
  return json.loads(out.read_text())


def _table(payload, name):
  return next(t for t in payload["tables"] if t["name"] == name)


def test_three_tables_with_shapes(payload):
  assert len(payload["tables"]) == 3
  for t in payload["tables"]:
    mp = t["minimal_pipeline"]
    T = t["n_train"] + t["n_test"]
    assert len(mp["x"]) == T
    assert all(len(row) == mp["d"] for row in mp["x"])
    assert len(mp["y_train"]) == mp["train_size"] == t["n_train"]
    assert len(mp["y_full_padded"]) == T
    assert mp["y_full_padded"][mp["train_size"]:] == \
        [-100.0] * (T - mp["train_size"])
    assert len(mp["cat_mask"]) == mp["d"]
    assert all(math.isfinite(v) for row in mp["x"] for v in row)


def test_first_appearance_and_min_frequency(payload):
  t = _table(payload, "mixed_types")
  cats = t["encoder"]["cat_categories_first_appearance"]
  # red appears before blue; green/violet are rare (1 occurrence) -> dropped
  assert cats["cat_c"] == ["red", "blue"]
  assert cats["bool_d"] == [True, False]
  # encoded train matrix: cat_c is column 0
  col0 = [row[0] for row in t["encoder"]["train_matrix"]]
  assert col0 == [0.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, -1.0]
  # unseen test category ('purple') and NULL -> -1
  test_col0 = [row[0] for row in t["encoder"]["test_matrix"]]
  assert test_col0[2] == -1.0 and test_col0[3] == -1.0


def test_label_encoding_alphabetical(payload):
  t = _table(payload, "mixed_types")
  assert t["target"]["classes_alphabetical"] == ["bird", "cat", "dog"]
  assert t["target"]["y_encoded"] == [1, 2, 0, 1, 2, 1, 0, 2]


def test_numeric_impute_mean(payload):
  t = _table(payload, "mixed_types")
  vals = [v for v in
          next(c for c in t["columns"] if c["name"] == "num_a")["values"][:8]
          if v is not None]
  assert t["encoder"]["numeric_impute_mean"]["num_a"] == \
      pytest.approx(sum(vals) / len(vals))


def test_datetime_expansion_and_fill(payload):
  t = _table(payload, "mixed_types")
  names = t["encoder"]["encoded_column_names"]
  assert names[-5:] == ["date_e.epoch_ns", "date_e.year", "date_e.month",
                        "date_e.day", "date_e.dayofweek"]
  fill = t["encoder"]["datetime_fill_train_mean"]["date_e"]
  assert fill["epoch_ns"] > 0
  # row 2 (train) has NaT -> filled with the train mean epoch
  epoch_col = names.index("date_e.epoch_ns")
  assert t["encoder"]["train_matrix"][2][epoch_col] == \
      pytest.approx(float(fill["epoch_ns"]))


def test_unique_filter_drops_constant(payload):
  t = _table(payload, "constant_and_rare")
  keep = t["minimal_pipeline"]["unique_filter_keep"]
  # encoded order: [cat_y, num_const, num_x] -> num_const dropped
  assert keep == [True, False, True]
  assert t["minimal_pipeline"]["d"] == 2
  assert t["minimal_pipeline"]["cat_feature_indices_kept"] == [0]
  assert t["minimal_pipeline"]["cat_mask"] == [True, False]


def test_regression_y_scaler(payload):
  t = _table(payload, "regression_dates")
  y = np.array(t["target"]["train_values"])
  sc = t["target"]["y_scaler"]
  assert sc["mean"] == pytest.approx(y.mean())
  assert sc["scale_std_ddof0"] == pytest.approx(y.std())
  got = np.array(t["target"]["y_scaled"])
  np.testing.assert_allclose(got, (y - y.mean()) / y.std(), rtol=1e-12)
  # string dates were detected as datetime
  assert "date_s" in t["encoder"]["column_classification"]["datetime"]


def test_zscore_stage_consistency(payload):
  t = _table(payload, "mixed_types")
  mp = t["minimal_pipeline"]
  enc_train = np.array(t["encoder"]["train_matrix"])
  keep = np.array(mp["unique_filter_keep"], dtype=bool)
  kept = enc_train[:, keep]
  np.testing.assert_allclose(np.array(mp["standard_scaler"]["mean"]),
                             kept.mean(axis=0), rtol=1e-9)
  np.testing.assert_allclose(
      np.array(mp["standard_scaler"]["scale_std_ddof0_plus_eps"]),
      kept.std(axis=0) + 1e-6, rtol=1e-9)


def test_ensemble_member_configs(payload):
  ens = payload["ensemble"]
  assert ens["seed"] == 42
  by_n = {c["n_estimators"]: c for c in ens["cases"]}
  one, four = by_n[1], by_n[4]
  H = one["n_features_after_filter"]

  m0 = one["members_flattened_order"][0]
  assert m0["feature_permutation"] == list(range(H))  # identity for n=1
  assert m0["class_shift_offset"] == 0
  assert m0["norm_method"] == "none"

  members = four["members_flattened_order"]
  assert len(members) == 4
  norms = [m["norm_method"] for m in members]
  # flattened order groups by norm method: none-members first, then power
  assert norms == sorted(norms, key=["none", "power"].index)
  assert set(norms) == {"none", "power"}
  for m in members:
    assert sorted(m["feature_permutation"]) == list(range(H))
    assert 0 <= m["class_shift_offset"] < four["n_classes"]
    assert m["d"] == H
    assert len(m["cat_mask"]) == H
    assert len(m["y_context_shifted"]) == 8
  # class shift actually varies across members for n_estimators=4
  assert len({m["class_shift_offset"] for m in members}) > 1


def test_nnls_blend(payload):
  b = payload["blend"]["nnls_classification"]
  w_raw = np.array(b["weights_raw"])
  assert (w_raw >= 0).all()
  w_final = np.array(b["weights_final"])
  assert w_final.sum() == pytest.approx(1.0)
  # recompute nnls from emitted inputs
  import scipy.optimize as opt
  probs = np.array(b["oof_probs_softmax_t0p9"])
  E = probs.shape[0]
  a = probs.reshape(E, -1).T
  ref, _ = opt.nnls(a, np.array(b["y_one_hot_flat"]))
  np.testing.assert_allclose(ref, w_raw, atol=1e-10)

  r = payload["blend"]["nnls_regression"]
  wr = np.array(r["weights_final"])
  assert wr.sum() == pytest.approx(1.0)
  np.testing.assert_allclose(
      np.array(r["blended_prediction"]),
      wr @ np.array(r["y_oof_members"]), rtol=1e-12)


def test_softmax_temperature(payload):
  s = payload["blend"]["softmax_temperature"]
  assert s["temperature"] == 0.9
  logits = np.array(s["logits"])
  z = logits / 0.9
  e = np.exp(z - z.max(axis=-1, keepdims=True))
  ref = e / e.sum(axis=-1, keepdims=True)
  np.testing.assert_allclose(np.array(s["probs"]), ref, rtol=1e-12)


def test_committed_golden_matches_regeneration(tmp_path):
  """If test/fixtures/golden_preprocess.json is committed, it must match a
  fresh regeneration byte-for-byte (deterministic generator)."""
  import pathlib
  committed = pathlib.Path(__file__).resolve().parents[3] / \
      "test/fixtures/golden_preprocess.json"
  if not committed.exists():
    pytest.skip("golden_preprocess.json not generated yet")
  fresh = tmp_path / "regen.json"
  golden_preprocess.generate(fresh)
  assert fresh.read_text() == committed.read_text()
