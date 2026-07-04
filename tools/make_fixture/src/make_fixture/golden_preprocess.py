"""Golden vectors for the C++ preprocessing/ensemble port (WS-F).

Everything in the emitted JSON is produced by the UNMODIFIED upstream code
(vendor/tabfm classifier_and_regressor.py, imported standalone):
CategoricalOrdinalEncoder / TransformToNumerical / DatetimeTransformer /
UniqueFeatureFilter / CustomStandardScaler / OutlierRemover /
EnsembleGenerator / TabFMClassifier.softmax / the NNLS blend recipe.

Structure of golden_preprocess.json (all fields documented in "_docs"):
  tables[]   3 mixed-type ~12-row tables and their exact encoded
             x/y/cat_mask/d/train_size tensors from the minimal pipeline
             (n_estimators=1, norm_methods=["none"], seed 42), plus every
             intermediate stage (encoded matrix, imputer means, category
             orders, datetime fills, unique filter, scaler, outlier bounds)
  ensemble   for seed 42 and n_estimators in {1,4}: the exact member
             configs the upstream EnsembleGenerator draws (feature
             permutation indices, class shift, scaler/norm choice), in the
             flattened prepare_ensemble_tensors order the runtime must use
  blend      NNLS blend-weight mechanism on toy cases (classification +
             regression) and the softmax_temperature=0.9 application
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import scipy.optimize as opt

from make_fixture import upstream

SEED = 42  # upstream _DEFAULT_RANDOM_STATE
SOFTMAX_TEMPERATURE = 0.9
NNLS_BETA = 0.75
PAD_SENTINEL = -100.0


# ---------------------------------------------------------------------------
# Test tables (12 rows = 8 train + 4 test; NULLs everywhere on purpose)
# ---------------------------------------------------------------------------

def _table_mixed_types():
  df = pd.DataFrame({
      "num_a": pd.array([1.5, None, 3.25, 4.0, 2.75, 5.5, 1.25, 3.0,
                         2.0, None, 4.5, 3.5], dtype="float64"),
      "num_b": pd.array([10, 12, 11, 15, 14, 13, 10, 12,
                         11, 14, 15, 10], dtype="int64"),
      "cat_c": pd.Series(["red", "blue", "red", "green", "blue", "red",
                          None, "violet", "green", "red", "purple", None],
                         dtype=object),
      "bool_d": pd.Series([True, False, True, None, False, True, True,
                           False, False, True, None, False], dtype=object),
      "date_e": pd.to_datetime([
          "2023-01-15", "2023-06-30", None, "2024-02-29", "2023-11-05",
          "2024-07-04", "2023-03-10", "2024-12-01",
          "2024-03-15", None, "2023-08-20", "2024-05-05"]),
  })
  y_train = ["cat", "dog", "bird", "cat", "dog", "cat", "bird", "dog"]
  return dict(name="mixed_types", df=df, n_train=8, task="classification",
              y_train=y_train,
              notes=[
                  "cat_c: train counts red=3 blue=2 green=1 violet=1 NULL=1;"
                  " min_cat_frequency=2 (TransformToNumerical default) keeps"
                  " only [red, blue] in first-appearance order; green/violet"
                  " (rare) and NULL and unseen test 'purple' all encode to"
                  " unknown_value=-1",
                  "bool_d: pandas OBJECT column (bools + NULL) -> categorical"
                  " path, categories in first-appearance order [True, False];"
                  " NULL -> -1. NOTE: a bool column WITHOUT nulls has pandas"
                  " dtype bool == numeric upstream (imputer path, cat_mask=0);"
                  " HLD 4.5 maps DuckDB BOOLEAN to categorical — the C++ port"
                  " should follow upstream on encoded values but be aware of"
                  " this upstream dtype quirk",
                  "date_e: datetime64 column with NaT in train and test; NaT"
                  " filled with the TRAIN mean timestamp learned at fit",
              ])


def _table_constant_and_rare():
  df = pd.DataFrame({
      "num_const": pd.array([7.0] * 12, dtype="float64"),
      "num_x": pd.array([0.5, 1.5, None, 2.5, 0.75, 1.25, 2.0, 0.25,
                         1.0, None, 2.25, 0.5], dtype="float64"),
      "cat_y": pd.Series(["A", "A", "B", "B", "A", "B", "A", "A",
                          "B", "A", "C", "B"], dtype=object),
  })
  y_train = ["no", "yes", "yes", "no", "no", "yes", "no", "no"]
  return dict(name="constant_and_rare", df=df, n_train=8,
              task="classification", y_train=y_train,
              notes=[
                  "num_const is constant on the 8 train rows ->"
                  " UniqueFeatureFilter drops it (encoded index 1); the"
                  " kept-column mask and the remapped cat_features_ index"
                  " are in unique_filter_keep / cat_feature_indices_kept",
                  "test category 'C' is unseen -> -1",
              ])


def _table_regression_dates():
  df = pd.DataFrame({
      "num_p": pd.array([10.0, 20.0, None, 40.0, 50.0, 60.0, 70.0, 80.0,
                         15.0, 25.0, None, 55.0], dtype="float64"),
      "num_q": pd.array([-1.0, 0.5, 1.5, -0.5, 2.0, 0.0, 1.0, -1.5,
                         0.25, -0.75, 1.75, 0.5], dtype="float64"),
      "cat_r": pd.Series(["x", "y", "x", "z", "y", "x", "z", "y",
                          "z", "x", "y", "w"], dtype=object),
      # date as STRINGS: exercises the _looks_like_datetime text-detection
      # path (pd.to_numeric fails -> to_datetime coerce parses)
      "date_s": pd.Series(["2024-01-05", "2024-02-10", None, "2024-04-20",
                           "2024-05-25", "2024-06-30", "2024-08-04",
                           "2024-09-09", "2024-10-14", None, "2024-12-24",
                           "2025-01-28"], dtype=object),
  })
  y_train = [100.0, 110.0, 95.0, 130.0, 140.0, 150.0, 160.0, 170.0]
  return dict(name="regression_dates", df=df, n_train=8, task="regression",
              y_train=y_train,
              notes=[
                  "date_s is a VARCHAR column of ISO dates -> upstream"
                  " detects it as datetime via the text heuristic; the C++"
                  " port receives DuckDB DATE/TIMESTAMP types directly and"
                  " should treat them as the datetime path",
                  "regression target is z-scored with sklearn"
                  " StandardScaler (ddof=0 std); mean/scale in y_scaler;"
                  " predictions are inverse-transformed",
              ])


def _jsonable_cell(v):
  if v is None or (isinstance(v, float) and np.isnan(v)):
    return None
  if isinstance(v, (np.bool_, bool)):
    return bool(v)
  if isinstance(v, (np.integer,)):
    return int(v)
  if isinstance(v, (np.floating,)):
    return float(v)
  if isinstance(v, pd.Timestamp):
    return v.isoformat()
  if v is pd.NaT:
    return None
  return v


def _df_to_json_columns(df: pd.DataFrame):
  cols = []
  for name in df.columns:
    s = df[name]
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
      typ = "date"
      vals = [None if pd.isna(v) else pd.Timestamp(v).date().isoformat()
              for v in s]
    elif pd.api.types.is_float_dtype(s.dtype):
      typ = "double"
      vals = [None if pd.isna(v) else float(v) for v in s]
    elif pd.api.types.is_integer_dtype(s.dtype):
      typ = "bigint"
      vals = [int(v) for v in s]
    else:
      # object: varchar / bool-with-null / date-strings
      typ = "varchar_or_bool"
      vals = [_jsonable_cell(v) for v in s]
    cols.append({"name": name, "duckdb_type_hint": typ, "values": vals})
  return cols


# ---------------------------------------------------------------------------
# Minimal pipeline (n_estimators=1, norm_methods=["none"]) per table
# ---------------------------------------------------------------------------

def _run_table(car, spec) -> dict:
  df, n_train, task = spec["df"], spec["n_train"], spec["task"]
  df_train, df_test = df.iloc[:n_train], df.iloc[n_train:]

  # --- target encoding (mirrors TabFMClassifier.fit / TabFMRegressor.fit)
  if task == "classification":
    y_encoder = car.CategoricalOrdinalEncoder(dtype=np.int64,
                                              mode="alphabetical")
    y_enc = y_encoder.fit_transform(
        np.array(spec["y_train"], dtype=object).reshape(-1, 1)).flatten()
    classes = [_jsonable_cell(c) for c in y_encoder.categories_[0]]
    n_classes = len(classes)
    target_block = {"task": task, "train_values": spec["y_train"],
                    "classes_alphabetical": classes,
                    "y_encoded": y_enc.tolist()}
  else:
    from sklearn.preprocessing import StandardScaler
    y_raw = np.asarray(spec["y_train"], dtype=np.float64)
    y_scaler = StandardScaler()
    y_enc = y_scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()
    n_classes = 0
    target_block = {"task": task, "train_values": spec["y_train"],
                    "y_scaler": {"mean": float(y_scaler.mean_[0]),
                                 "scale_std_ddof0": float(y_scaler.scale_[0])},
                    "y_scaled": y_enc.tolist()}

  # --- feature encoding (TransformToNumerical: appearance order,
  # min_cat_frequency=2, mean-impute numerics, datetime expansion)
  x_encoder = car.TransformToNumerical(cat_encoder_mode="appearance")
  X_train = x_encoder.fit_transform(df_train)
  X_test = x_encoder.transform(df_test)

  tfm = x_encoder.tfm_
  cat_pos = list(tfm.transformers_[0][2])
  num_pos = list(tfm.transformers_[1][2])
  dt_pos = list(tfm.transformers_[2][2])
  cat_cols = [df.columns[i] for i in cat_pos]
  num_cols = [df.columns[i] for i in num_pos]
  dt_cols = [df.columns[i] for i in dt_pos]

  cat_enc = tfm.named_transformers_["categorical"]
  imputer = tfm.named_transformers_["continuous"]
  dt_tfm = tfm.named_transformers_["datetime"]

  encoded_names = list(cat_cols) + list(num_cols)
  for c in dt_cols:
    encoded_names.append(f"{c}.epoch_ns")
    for f in dt_tfm.features:
      encoded_names.append(f"{c}.{f}")

  encoder_block = {
      "column_classification": {"categorical": cat_cols, "numeric": num_cols,
                                "datetime": dt_cols},
      "encoded_column_names": encoded_names,
      "cat_categories_first_appearance": {
          c: [_jsonable_cell(v) for v in cat_enc.categories_[i]]
          for i, c in enumerate(cat_cols)},
      "cat_unknown_value": -1,
      "min_cat_frequency": 2,
      "numeric_impute_mean": {c: float(imputer.statistics_[i])
                              for i, c in enumerate(num_cols)},
      "datetime_fill_train_mean": {
          c: {"iso": pd.Timestamp(dt_tfm._fillna_map[c]).isoformat(),
              "epoch_ns": int(pd.Timestamp(dt_tfm._fillna_map[c]).value)}
          for c in dt_cols},
      "datetime_derived_features": (["epoch_ns"] + dt_tfm.features
                                    if dt_cols else []),
      "train_matrix": X_train.tolist(),
      "test_matrix": X_test.tolist(),
  }

  # --- minimal ensemble (n_estimators=1, norm none) — exact model tensors
  cat_features = list(range(len(cat_pos)))
  eg = car.EnsembleGenerator(
      n_estimators=1, norm_methods=["none"], feat_shuffle_method="random",
      class_shift=(task == "classification"), cat_features=cat_features,
      outlier_threshold=4.0, max_num_features=500, random_state=SEED,
      task=task)
  eg.fit(X_train, y_enc)
  data = eg.transform(X_test)
  Xs, ys, cat_masks, ds, configs = eg.prepare_ensemble_tensors(data)
  assert Xs.shape[0] == 1
  shuffle_pattern, shift_offset, cat_perm, row_sub = configs[0]
  assert shift_offset == 0 and cat_perm is None and row_sub is None

  pre = eg.preprocessors_["none"]
  T = Xs.shape[1]
  y_train_ctx = ys[0]
  y_full = np.full(T, PAD_SENTINEL)
  y_full[: len(y_train_ctx)] = y_train_ctx

  minimal = {
      "seed": SEED,
      "n_estimators": 1,
      "norm_method": "none",
      "unique_filter_keep": eg.unique_filter_.features_to_keep_.tolist(),
      "cat_feature_indices_kept": eg.cat_features_.tolist(),
      "feature_permutation": np.asarray(shuffle_pattern).tolist(),
      "standard_scaler": {
          "mean": pre.standard_scaler_.mean_.tolist(),
          "scale_std_ddof0_plus_eps": pre.standard_scaler_.scale_.tolist(),
          "epsilon": 1e-6, "clip_min": -100.0, "clip_max": 100.0},
      "outlier_remover": {
          "threshold": 4.0,
          "means": pre.outlier_remover_.means_.tolist(),
          "stds": pre.outlier_remover_.stds_.tolist(),
          "lower_bounds": pre.outlier_remover_.lower_bounds_.tolist(),
          "upper_bounds": pre.outlier_remover_.upper_bounds_.tolist()},
      "x": Xs[0].tolist(),
      "y_train": y_train_ctx.tolist(),
      "y_full_padded": y_full.tolist(),
      "train_size": int(len(y_train_ctx)),
      "cat_mask": cat_masks[0].astype(bool).tolist(),
      "d": int(ds[0]),
  }
  return {
      "name": spec["name"],
      "notes": spec["notes"],
      "n_train": n_train,
      "n_test": len(df) - n_train,
      "columns": _df_to_json_columns(df),
      "target": target_block,
      "encoder": encoder_block,
      "minimal_pipeline": minimal,
  }


# ---------------------------------------------------------------------------
# Ensemble member configs for seed 42, n_estimators in {1, 4}
# ---------------------------------------------------------------------------

def _run_ensemble_cases(car, spec) -> dict:
  df, n_train, task = spec["df"], spec["n_train"], spec["task"]
  df_train = df.iloc[:n_train]
  y_encoder = car.CategoricalOrdinalEncoder(dtype=np.int64,
                                            mode="alphabetical")
  y_enc = y_encoder.fit_transform(
      np.array(spec["y_train"], dtype=object).reshape(-1, 1)).flatten()
  x_encoder = car.TransformToNumerical(cat_encoder_mode="appearance")
  X_train = x_encoder.fit_transform(df_train)
  n_cat = len(x_encoder.tfm_.transformers_[0][2])

  cases = []
  for n_estimators in (1, 4):
    eg = car.EnsembleGenerator(
        n_estimators=n_estimators, norm_methods=None,  # default none+power
        feat_shuffle_method="random", class_shift=True,
        cat_features=list(range(n_cat)), outlier_threshold=4.0,
        max_num_features=500, random_state=SEED, task=task)
    eg.fit(X_train, y_enc)
    data = eg.transform(x_encoder.transform(df.iloc[n_train:]))
    Xs, ys, cat_masks, ds, configs = eg.prepare_ensemble_tensors(data)

    norm_by_member = []
    for norm_method, cfgs in eg.ensemble_configs_.items():
      norm_by_member.extend([norm_method] * len(cfgs))

    members = []
    for i, (pattern, shift, cat_perm, row_sub) in enumerate(configs):
      members.append({
          "member_index_flat": i,
          "norm_method": norm_by_member[i],
          "feature_permutation": np.asarray(pattern).tolist(),
          "class_shift_offset": int(shift),
          "cat_permutation": None if cat_perm is None else "unused",
          "row_subsample": None if row_sub is None
          else np.asarray(row_sub).tolist(),
          "d": int(ds[i]),
          "cat_mask": cat_masks[i].astype(bool).tolist(),
          "y_context_shifted": ys[i].tolist(),
      })
    cases.append({
        "n_estimators": n_estimators,
        "table": spec["name"],
        "n_features_after_filter": int(eg.n_features_in_),
        "n_classes": int(eg.n_classes_),
        "norm_methods_default": ["none", "power"],
        "members_flattened_order": members,
    })
  return {"seed": SEED, "cases": cases}


# ---------------------------------------------------------------------------
# Blend: NNLS weights + softmax temperature
# ---------------------------------------------------------------------------

def _run_blend(car) -> dict:
  softmax = car.TabFMClassifier.softmax

  # softmax_temperature application (upstream: x/T, max-subtract, exp, norm)
  logits_example = np.array([[2.0, 1.0, 0.5],
                             [-1.0, 0.0, 1.0],
                             [0.3, 0.3, 0.3]])
  probs_example = softmax(logits_example, axis=-1,
                          temperature=SOFTMAX_TEMPERATURE)

  # classification NNLS toy (mirrors TabFMClassifier.fit enable_nnls path)
  rng = np.random.default_rng(SEED)
  E, N, K = 4, 6, 3
  oof_logits = rng.normal(size=(E, N, K)).round(6)
  oof_probs = softmax(oof_logits, axis=-1, temperature=SOFTMAX_TEMPERATURE)
  y = np.array([0, 1, 2, 0, 1, 1])
  y_one_hot = np.zeros((N, K))
  y_one_hot[np.arange(N), y] = 1.0
  oof_flat = oof_probs.reshape(E, N * K)
  w_raw, rnorm = opt.nnls(oof_flat.T, y_one_hot.flatten())
  w_norm = w_raw / w_raw.sum() if w_raw.sum() > 0 else np.ones(E) / E
  avg = np.ones(E) / E
  w_final = NNLS_BETA * w_norm + (1.0 - NNLS_BETA) * avg
  probs_blend = np.tensordot(w_final, oof_probs, axes=(0, 0))

  # regression NNLS toy (mirrors TabFMRegressor.fit enable_nnls path;
  # inputs are already inverse-transformed member predictions)
  E_r, N_r = 3, 6
  y_orig = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  y_oof = np.stack([
      y_orig + rng.normal(scale=0.3, size=N_r).round(6),
      y_orig * 1.1 + 0.2,
      np.full(N_r, y_orig.mean()),
  ])
  wr_raw, _ = opt.nnls(y_oof.T, y_orig)
  wr_norm = wr_raw / wr_raw.sum() if wr_raw.sum() > 0 else np.ones(E_r) / E_r
  avg_r = np.ones(E_r) / E_r
  wr_final = NNLS_BETA * wr_norm + (1.0 - NNLS_BETA) * avg_r
  pred_blend = wr_final @ y_oof

  return {
      "softmax_temperature": {
          "temperature": SOFTMAX_TEMPERATURE,
          "formula": "softmax(logits / temperature) with max-subtraction "
                     "for stability (upstream TabFMClassifier.softmax)",
          "logits": logits_example.tolist(),
          "probs": probs_example.tolist(),
      },
      "nnls_classification": {
          "formula": "w_raw = nnls(oof_probs.reshape(E, N*K).T, "
                     "one_hot(y).flatten()); w = w_raw/sum(w_raw) (uniform "
                     "if sum==0); final = nnls_beta*w + (1-nnls_beta)/E; "
                     "blended probs = tensordot(final, probs_all, (0,0))",
          "nnls_beta": NNLS_BETA,
          "oof_logits": oof_logits.tolist(),
          "oof_probs_softmax_t0p9": oof_probs.tolist(),
          "y": y.tolist(),
          "y_one_hot_flat": y_one_hot.flatten().tolist(),
          "weights_raw": w_raw.tolist(),
          "nnls_residual_norm": float(rnorm),
          "weights_normalized": w_norm.tolist(),
          "weights_final": w_final.tolist(),
          "blended_probs": probs_blend.tolist(),
      },
      "nnls_regression": {
          "formula": "w_raw = nnls(y_oof.T, y_orig) on inverse-transformed "
                     "member predictions; normalize; blend with uniform via "
                     "nnls_beta; final prediction = final_weights @ member_"
                     "predictions (inverse-transformed)",
          "nnls_beta": NNLS_BETA,
          "y_orig": y_orig.tolist(),
          "y_oof_members": y_oof.tolist(),
          "weights_raw": wr_raw.tolist(),
          "weights_normalized": wr_norm.tolist(),
          "weights_final": wr_final.tolist(),
          "blended_prediction": pred_blend.tolist(),
      },
  }


_DOCS = {
    "tables[].columns": "raw column values as a DuckDB table would hold "
        "them; null == SQL NULL. duckdb_type_hint maps to the HLD 4.5 type "
        "table.",
    "tables[].target": "classification: labels are encoded with "
        "CategoricalOrdinalEncoder(mode=alphabetical) — classes sorted "
        "ascending, y_encoded are their indices. regression: y z-scored "
        "with StandardScaler (mean, std ddof=0).",
    "tables[].encoder.column_classification": "upstream column typing: "
        "datetime64 dtype or text-that-parses-as-dates -> datetime; other "
        "numeric dtypes -> numeric; everything else (incl. object bool "
        "with NULLs) -> categorical.",
    "tables[].encoder.encoded_column_names": "column order of the encoded "
        "matrix: ALL categorical (ordinal codes) first, then numeric "
        "(mean-imputed), then per datetime column [epoch_ns, year, month, "
        "day, dayofweek]. epoch_ns is int64 nanoseconds cast to float64 "
        "(precision loss above 2^53 is upstream behavior the port must "
        "replicate by casting int64 ns -> double).",
    "tables[].encoder.cat_categories_first_appearance": "kept categories "
        "per column in FIRST-APPEARANCE order over the train rows; "
        "categories with < min_cat_frequency occurrences, NULLs and unseen "
        "test values all encode to -1.",
    "tables[].encoder.datetime_fill_train_mean": "NaT/unparseable dates "
        "are replaced by the TRAIN mean timestamp (learned at fit, reused "
        "at transform).",
    "tables[].minimal_pipeline": "EnsembleGenerator(n_estimators=1, "
        "norm_methods=['none'], random_state=42) on the encoded matrix: "
        "UniqueFeatureFilter (drop <=1 unique value on train) -> "
        "CustomStandardScaler (z-score with std(ddof=0)+1e-6, clip to "
        "[-100,100]) -> OutlierRemover (two-stage 4-sigma log clipping). "
        "x = [train; test] rows after that chain with the identity feature "
        "permutation; y_train = encoded/scaled train labels; y_full_padded "
        "pads to T with -100.0; cat_mask marks kept categorical columns; "
        "d = active feature count. Cast x/y to float32 before the model "
        "feed (tensors here are the float64 pipeline outputs).",
    "tables[].minimal_pipeline.outlier_remover": "transform clips as "
        "X = max(X, -log1p(|X|) + lower) then X = min(X, log1p(|X|) + "
        "upper); bounds/means/stds are the second-stage (outliers-removed) "
        "statistics from fit.",
    "ensemble.cases[].members_flattened_order": "the EXACT member configs "
        "EnsembleGenerator draws for seed 42, in prepare_ensemble_tensors' "
        "flattened order (grouped by norm method: all 'none' members "
        "first, then 'power'). feature_permutation indexes the "
        "unique-filtered encoded columns; x_member = x[:, permutation]. "
        "class_shift_offset: y_context = (y_enc + offset) % n_classes; "
        "after the forward pass, un-shift the logits with out = "
        "concat(out[..., offset:], out[..., :offset]). norm_method is the "
        "scaler choice ('none' -> minimal chain above, 'power' -> "
        "PowerTransformer(yeo-johnson) inserted between scaler and outlier "
        "stage). Member x tensors for 'power' are NOT emitted (sklearn-"
        "version-sensitive); assert configs, not transforms, for those.",
    "blend.nnls_*": "NNLS blend fitting (enable_nnls preset): weights from "
        "scipy nnls on out-of-fold predictions, normalized, then blended "
        "with the uniform vector via nnls_beta=0.75.",
    "blend.softmax_temperature": "temperature 0.9 softmax applied to "
        "per-member logits BEFORE probability averaging / NNLS blending "
        "(classification).",
}


def generate(out_path: pathlib.Path) -> dict:
  car = upstream.load()
  specs = [_table_mixed_types(), _table_constant_and_rare(),
           _table_regression_dates()]
  payload = {
      "_docs": _DOCS,
      "generator": {
          "tool": "tools/make_fixture (golden_preprocess)",
          "seed": SEED,
          "upstream": "vendor/tabfm/tabfm/src/classifier_and_regressor.py "
                      "(unmodified, imported standalone)",
          "versions": _versions(),
      },
      "tables": [_run_table(car, s) for s in specs],
      "ensemble": _run_ensemble_cases(car, specs[0]),
      "blend": _run_blend(car),
  }
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(payload, indent=1) + "\n")
  return payload


def _versions() -> dict:
  import scipy
  import sklearn
  return {"numpy": np.__version__, "pandas": pd.__version__,
          "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}
