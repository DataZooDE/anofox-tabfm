# Real-model verification scenarios

Two end-to-end, **pure-SQL** scenarios that exercise the real TabFM v1 model
(the full 6.56 GB weights) against public Hugging Face datasets, read directly
over `hf://datasets/...`. Each splits the data into an in-context ("train")
set and a scored ("test") set, predicts with `tabfm_classify` /
`tabfm_regress`, and computes the metric entirely in SQL.

## Running

```bash
LOAD anofox_tabfm; INSTALL httpfs;
SET anofox_tabfm_accept_hf_license = true;
CALL tabfm_download('classification');   -- 6.56 GB, once
CALL tabfm_download('regression');       -- once

duckdb :memory: < scenarios/churn_f1.sql    # classification, F1
duckdb :memory: < scenarios/tips_mse.sql    # regression, MSE
```

Each scenario points `anofox_tabfm_model_manifest` at the matching
`scenarios/tabfm_real_<task>.json`, which references the weight-free graph in
`resources/` and resolves the downloaded weights from the cache. The graphs
ship weight-free; the weights are the user's own download (license-clean).

## Results (single 8-core x86 CPU, fp32, n_estimators = 1)

**Churn classification** — `scikit-learn/churn-prediction`, 500 in-context
rows, 150 scored, target `Churn`:

| metric | value |
|---|---|
| accuracy | 0.827 |
| precision (churn) | 0.765 |
| recall (churn) | 0.591 |
| **F1 (churn)** | **0.667** |
| confusion (tp/fp/fn/tn) | 26 / 8 / 18 / 98 |
| wall time | ~63 s (incl. one-time 6.6 GB model load) |

**Tips regression** — `scikit-learn/tips`, ~180 in-context rows, 56 scored,
target `tip`:

| metric | value |
|---|---|
| **MSE** | **0.971** |
| RMSE | 0.986 |
| MAE | 0.730 |
| mean-predictor baseline MSE | 1.680 |
| wall time | ~54 s |

Zero-shot, no training: the model reads the train split as context and scores
the test split. Classification reaches 0.67 F1 on churn; regression beats the
mean-predictor baseline by ~42 % on tips. These are single-estimator results;
the ensemble path (`n_estimators > 1`, M3) is expected to improve them.

## Notes

- The join-back to the held-out labels uses a key column carried through the
  prediction (`customerID` / a synthetic `row_id`); as an unseen categorical
  in the test split it is inert and does not leak.
- Determinism: same inputs + same seed → identical predictions and metrics.
