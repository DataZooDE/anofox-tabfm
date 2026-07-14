# Real-model verification examples

Five end-to-end, **pure-SQL** examples that exercise the real TabFM v1 model
(the full 6.56 GB weights) against public Hugging Face datasets, read directly
over `hf://datasets/...`. Each splits the data into an in-context ("train") set
and a scored ("test") set, predicts with `tabfm_classify` / `tabfm_regress`, and
computes the metric entirely in SQL.

| example | task | dataset | metric |
|---|---|---|---|
| [`classification_churn.sql`](classification_churn.sql) | binary classification | `scikit-learn/churn-prediction` | F1 |
| [`classification_iris.sql`](classification_iris.sql) | 3-class classification | `scikit-learn/iris` | accuracy |
| [`classification_income.sql`](classification_income.sql) | binary classification | `scikit-learn/adult-census-income` | F1 (also shows `tabfm_list_models()` + `model :=`) |
| [`regression_tips.sql`](regression_tips.sql) | regression | `scikit-learn/tips` | MSE |
| [`regression_wine.sql`](regression_wine.sql) | regression | `mstz/wine` | MSE |
| [`compare_models.sql`](compare_models.sql) | **multi-model** | `scikit-learn/iris` | accuracy **+ runtime**, two models |

## Running

```bash
LOAD anofox_tabfm; INSTALL httpfs;
SET anofox_tabfm_accept_hf_license = true;
CALL tabfm_download('classification');   -- 6.56 GB, once (churn + iris)
CALL tabfm_download('regression');       -- once (tips)

# run from the repository root:
duckdb :memory: < examples/classification_churn.sql   # binary, F1
duckdb :memory: < examples/classification_iris.sql    # 3-class, accuracy
duckdb :memory: < examples/classification_income.sql  # binary, F1 (+ registry)
duckdb :memory: < examples/regression_tips.sql        # regression, MSE
duckdb :memory: < examples/regression_wine.sql        # regression, MSE
duckdb :memory: < examples/compare_models.sql         # multi-model: accuracy + runtime
```

Each example points `anofox_tabfm_model_manifest` at the matching
`examples/tabfm_real_<task>.json`, which references the weight-free graph in
`resources/` (relative to the manifest) and resolves the downloaded weights from
the cache. The graphs ship weight-free; the weights are the user's own download
(license-clean).

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

**Iris classification** — `scikit-learn/iris`, ~100 in-context rows, 53 scored,
3-class target `species`:

| metric | value |
|---|---|
| **accuracy** | **0.943** (50 / 53) |
| recall — setosa | 1.000 (17 / 17) |
| recall — versicolor | 0.938 (15 / 16) |
| recall — virginica | 0.900 (18 / 20) |
| wall time | ~41 s (incl. one-time 6.6 GB model load) |

**Adult-income classification** — `scikit-learn/adult-census-income`, 600
in-context rows, 200 scored, target `income` (`>50K` vs `<=50K`); also prints the
registry via `tabfm_list_models()` and selects `model := 'tabfm-v1'`:

| metric | value |
|---|---|
| accuracy | 0.860 |
| precision (>50K) | 0.733 |
| recall (>50K) | 0.673 |
| **F1 (>50K)** | **0.702** |
| confusion (tp/fp/fn/tn) | 33 / 12 / 16 / 139 |

**Tips regression** — `scikit-learn/tips`, ~180 in-context rows, 56 scored,
target `tip`:

| metric | value |
|---|---|
| **MSE** | **0.971** |
| RMSE | 0.986 |
| MAE | 0.730 |
| mean-predictor baseline MSE | 1.680 |
| wall time | ~54 s |

**Wine-quality regression** — `mstz/wine`, 200 in-context rows, 64 scored,
target `quality`:

| metric | value |
|---|---|
| **MSE** | **0.498** |
| RMSE | 0.706 |
| MAE | 0.529 |
| mean-predictor baseline MSE | 0.860 |
| wall time | ~42 s |

**Multi-model comparison** — `compare_models.sql`, `scikit-learn/iris`, the same
100-row context / 53 scored split run through **FOUR REAL foundation models**
selected with `model :=` (accuracy *and* wall-clock runtime side by side):

| model | license | size | accuracy | runtime |
|---|---|---|---|---|
| **`mitra`** (AWS Mitra) | Apache-2.0 (commercial) | ~303 MB | **0.962** | ~2.4 s |
| **`tabpfn-v2`** (Prior Labs) | Prior Labs (comm. + attrib.) | **~29 MB** | **0.962** | **~0.5 s** |
| **`tabicl-v2`** (Inria) | BSD-3 (commercial) | ~110 MB | **0.962** | ~0.7 s |
| `tabfm-v1` (Google) | non-commercial (gated) | 6.56 GB | 0.943 | ~30 s |

The three permissively-licensed foundation models **beat** the 1.6 B-param gated
Google model on iris, at a tiny fraction of the size and 40–60× the speed — and
comparing them is just `model :=` over one registry (`tabfm_list_models()` to
discover). All four also do **regression** (Mitra, TabPFN v2, TabICL v2, TabFM);
on `mstz/wine` TabPFN scores MSE 0.482 and TabICL 0.586 vs a 0.860 mean baseline.
Runtime is the per-predict `Run Time (s)` from `.timer on`.

Setup (all one-time): Mitra downloads from HF (`CALL tabfm_download('classification')`
under `examples/mitra.json`, ~303 MB, ungated); TabPFN v2 and TabICL v2 ship
PyTorch `.ckpt` (pickle), so a one-time converter stages real weights as
safetensors (`uv run python tools/export_{tabpfn,tabicl}/convert_weights.py
classification ~/.cache/anofox-tabfm`); TabFM is the cached 6.56 GB download. See
[`../docs/REAL_MODELS.md`](../docs/REAL_MODELS.md) for the full story (why Mitra is
a manifest-only drop-in while TabPFN/TabICL needed the y-as-train-prefix engine
feature). `compare_models.sql` itself runs the `mitra` vs `tabfm-v1` pair (both
HF-downloadable, no converter needed).

Zero-shot, no training: the model reads the train split as context and scores
the test split. Classification reaches 0.67 F1 on churn and 0.94 accuracy on
iris; regression beats the mean-predictor baseline by ~42 % on tips. These are
single-estimator results; the ensemble path (`n_estimators > 1`, M3) is expected
to improve them.

## Notes

- The join-back to the held-out labels uses a key column carried through the
  prediction (`customerID` / a synthetic `row_id`); as an unseen categorical in
  the test split it is inert and does not leak.
- Determinism: same inputs + same seed → identical predictions and metrics.
