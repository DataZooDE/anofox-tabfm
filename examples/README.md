# Real-model examples

End-to-end, **pure-SQL** examples exercising the real models. Two groups:
**prediction** (classify / regress against public Hugging Face datasets, read
directly over `hf://datasets/...`, scored entirely in SQL) and **generation**
(synthesize rows and fill gaps, on locally generated tables so they need no
download beyond the weights).

### Prediction

| example | task | dataset | metric |
|---|---|---|---|
| [`classification_churn.sql`](classification_churn.sql) | binary classification | `scikit-learn/churn-prediction` | F1 |
| [`classification_iris.sql`](classification_iris.sql) | 3-class classification | `scikit-learn/iris` | accuracy |
| [`classification_income.sql`](classification_income.sql) | binary classification | `scikit-learn/adult-census-income` | F1 (also shows `tabfm_list_models()` + `model :=`) |
| [`regression_tips.sql`](regression_tips.sql) | regression | `scikit-learn/tips` | MSE |
| [`regression_wine.sql`](regression_wine.sql) | regression | `mstz/wine` | MSE |
| [`compare_models.sql`](compare_models.sql) | **multi-model** | `scikit-learn/iris` | accuracy **+ runtime**, two models |

### Generation and imputation

| example | what it shows |
|---|---|
| [`generate_synthetic.sql`](generate_synthetic.sql) | `tabfm_generate`: sample new rows, the `temperature` knob, the insert-back round trip, seed reproducibility |
| [`impute_missing.sql`](impute_missing.sql) | `tabfm_impute`: fill NULLs, prove known cells are untouched, score the fills against held-out truth, `columns :=` and `rounds` |
| [`generate_fidelity.sql`](generate_fidelity.sql) | **check the output instead of trusting it** — marginals, class shares, and the correlation test that independent per-column sampling would fail |
| [`generate_conditional.sql`](generate_conditional.sql) | conditional generation: filter the source relation, or complete partially-specified rows |
| [`generate_breast_cancer.sql`](generate_breast_cancer.sql) | **the benchmark** — reproduces the [Prior Labs cookbook](https://docs.priorlabs.ai/cookbook/generate_synthetic_data) on the breast-cancer dataset: 30 features, 2× sample count, marginals + 435-pair correlation structure + a train-on-synthetic/test-on-real utility score |

## Running

```bash
LOAD anofox_tabfm; INSTALL httpfs;
SET anofox_tabfm_accept_hf_license = true;
CALL tabfm_download('classification', model := 'tabfm-v1');   -- 6.56 GB, once (churn + iris)
CALL tabfm_download('regression', model := 'tabfm-v1');       -- once (tips)

# run from the repository root:
duckdb :memory: < examples/classification_churn.sql   # binary, F1
duckdb :memory: < examples/classification_iris.sql    # 3-class, accuracy
duckdb :memory: < examples/classification_income.sql  # binary, F1 (+ registry)
duckdb :memory: < examples/regression_tips.sql        # regression, MSE
duckdb :memory: < examples/regression_wine.sql        # regression, MSE
duckdb :memory: < examples/compare_models.sql         # multi-model: accuracy + runtime

duckdb :memory: < examples/generate_synthetic.sql     # synthesize rows
duckdb :memory: < examples/impute_missing.sql         # fill NULLs (needs both tasks)
duckdb :memory: < examples/generate_fidelity.sql      # marginals + correlation checks
duckdb :memory: < examples/generate_conditional.sql   # constrained generation
```

The generation examples need only the **classification** weights, except
`impute_missing.sql` and part of `generate_conditional.sql`, which fill numeric
columns and therefore also need the regression weights. `tabfm_generate` never
needs regression at all — every chain-rule step is a classification problem (see
[`docs/GENERATE.md`](../docs/GENERATE.md)).

The signatures these examples use are covered offline by
`test/sql/tabfm_examples.test`, which runs them against the CI fixture model, so
a signature change breaks the build rather than the docs.

Every model here is **built in** — `tabfm-v1`, `mitra`, `tabpfn-v2`, `tabicl-v2`
are usable by name (`model := '<id>'`) with no manifest file; `tabfm_list_models()`
lists them. The weight-free graphs ship inside the extension; the weights are the
user's own download (license-clean). To register your *own* model in SQL, see
`CALL tabfm_register_model(...)`.

## Synthetic-data results — the Prior Labs cookbook, reproduced

`generate_breast_cancer.sql`, run with **`mitra`** on CPU. Breast-cancer
(Wisconsin) as upstream uses it: 389 training rows, 30 features, 762 synthetic
rows generated (2× the training split, as upstream), `temperature = 1.0`,
`seed = 42`. 29 sequential model calls, ~15 min.

**Utility — the test that matters.** Classify `diagnosis` on the *same* 180
held-out real rows, changing only what the model gets as in-context examples:

| in-context examples | accuracy |
|---|---|
| the real training rows | **98.33%** |
| **762 synthetic rows** | **97.78%** |

A model given only data that never existed lands within **0.55 points** of one
given the real thing.

**Fidelity:**

| check | value |
|---|---|
| mean marginal shift (in units of each feature's own SD) | 0.033 |
| worst marginal shift, over all 30 features | 0.096 |
| average SD error | 0.052 |
| correlation of correlations, over all 435 feature pairs | **0.970** |
| mean absolute correlation error | 0.096 |
| class balance (real → synthetic) | 63.8/36.2 → 63.6/36.4 |
| synthetic rows identical to a real row | **0** |

**Known limitation, visible in the numbers.** Correlations are *attenuated*: the
strongest real pair (`radius_mean` / `perimeter_mean`, r = 0.998) comes back at
0.954. This is the quantile binning — 10 bins cannot express r = 0.998 — and it
is not fixed by better within-bin sampling. Exact bar-distribution sampling is
the real fix; see [`docs/GENERATE.md`](../docs/GENERATE.md).

## Prediction results (single 8-core x86 CPU, fp32, n_estimators = 1)

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

Setup (all one-time): Mitra downloads from HF (`CALL tabfm_download('classification', model := 'mitra')`
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
