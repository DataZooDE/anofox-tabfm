# Getting Started

Install the extension, download a model's weights and make a first zero-shot prediction on the iris
dataset.

## 1. Install and load

```sql
INSTALL httpfs;          -- weights are fetched over HTTPS. Install it FIRST, before
LOAD httpfs;             -- any custom repository is set (see the note below).

INSTALL anofox_tabfm FROM community;
LOAD anofox_tabfm;
```

The community build is signed, so no extra flags are needed — but it is
published by a separate submission and can trail a release by days. To take a
release the moment it is out, use the anofox repository instead. Those builds
are **unsigned**, so DuckDB must be started with `-unsigned`:

```bash
duckdb -unsigned
```
```sql
INSTALL httpfs; LOAD httpfs;   -- FIRST: see below
SET custom_extension_repository = 'https://get.anofox.com';
INSTALL anofox_tabfm;
LOAD anofox_tabfm;
```

Two things that are easy to hit and give unhelpful errors:

* Without `-unsigned`, installing from `get.anofox.com` fails with *"Attempting
  to install an extension file that doesn't have a valid signature"*. The
  community repository needs no flag.
* `custom_extension_repository` applies to **every** install, including ones
  DuckDB triggers for you. If it is set before `httpfs` is present, DuckDB
  autoloads `httpfs` from `get.anofox.com`, which does not serve it, and the
  failure names `httpfs` rather than the setting. Install `httpfs` first.

Check which build you got:

```sql
SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'anofox_tabfm';
```

## 2. Pick a model and download its weights (once)

`SELECT model, capabilities, license, commercial, downloaded FROM tabfm_list_models();` lists every
built-in model ([the table](../../README.md#models) says the same). `mitra` is a good first model:
Apache-2.0, ungated, about 303 MB per task. The extension ships only **weight-free** graphs; the weights
are downloaded from Hugging Face before the model is used:

```sql
INSTALL httpfs; LOAD httpfs;
CALL tabfm_download('classification', model := 'mitra');    -- ~303 MB, cached in ~/.cache/anofox-tabfm
CALL tabfm_download('regression', model := 'mitra');        -- only if you will call tabfm_regress
```

Downloading does not select a model. Name it in every call with `model := 'mitra'`, or set it once for
the session:

```sql
SET anofox_tabfm_default_model = 'mitra';
```

Without either, a call fails with *"N models are registered (...) and none is selected"* and names both
remedies.

### Licence acknowledgement and gated repositories

Two separate gates can stand between you and a model's weights:

* **Licence acknowledgement** (this extension's gate). A model whose *Gated* column in
  [the Models table](../../README.md#models) says *yes* downloads only after you acknowledge its licence:

  ```sql
  SET anofox_tabfm_accept_hf_license = true;
  CALL tabfm_download('classification', model := 'tabpfn-v2');
  ```

  Without it `tabfm_download` fails and prints that `SET` statement. The setting covers every gated model,
  whatever its licence; read the licence in the model's Hugging Face repository first. The acceptance is
  recorded in the cache on the first gated download.
* **Gated Hugging Face repository** (Hugging Face's gate). Some repositories serve their files only to a
  signed-in account that has accepted the licence on the model page. Accept it there, then pass your token
  with a standard DuckDB secret:

  ```sql
  CREATE SECRET hf (TYPE http, BEARER_TOKEN 'hf_xxx', SCOPE 'https://huggingface.co');
  ```

  A download that needs this fails with HTTP 401 (no token) or 403 (token fine, licence not accepted on the
  model page), and the error names both steps. See
  [`docs/REAL_MODELS.md`](../REAL_MODELS.md#gated-huggingface-repositories).

A model can need both.

## 3. Predict

A concrete example — the classic **iris** dataset (classifying four flower measurements into
species), read straight from Hugging Face, with 20% of the species hidden as
`NULL` so we can predict them:

```sql
CREATE TABLE iris AS
SELECT SepalLengthCm AS sepal_len, SepalWidthCm AS sepal_wid,
       PetalLengthCm AS petal_len, PetalWidthCm AS petal_wid,
       CASE WHEN row_number() OVER () % 5 = 0 THEN NULL ELSE Species END AS species
FROM 'hf://datasets/scikit-learn/iris/**/*.csv';
```

Rows with a known `species` are the **context**; the `NULL` rows are the ones to
score:

| sepal_len | sepal_wid | petal_len | petal_wid | species |
|--:|--:|--:|--:|:--|
| 4.9 | 3.0 | 1.4 | 0.2 | Iris-setosa |
| 5.0 | 3.6 | 1.4 | 0.2 | **NULL** |
| 5.2 | 2.7 | 3.9 | 1.4 | **NULL** |
| 6.5 | 3.0 | 5.8 | 2.2 | **NULL** |

Predict the hidden species — zero-shot, no training:

```sql
SELECT sepal_len, sepal_wid, petal_len, petal_wid, yhat AS predicted_species, yhat_score
FROM tabfm_classify('iris', 'species', model := 'mitra')
WHERE species IS NULL;
```

| sepal_len | sepal_wid | petal_len | petal_wid | predicted_species | yhat_score |
|--:|--:|--:|--:|:--|--:|
| 5.0 | 3.6 | 1.4 | 0.2 | Iris-setosa | 1.00 |
| 5.2 | 2.7 | 3.9 | 1.4 | Iris-versicolor | 1.00 |
| 6.5 | 3.0 | 5.8 | 2.2 | Iris-virginica | 1.00 |

That's it — a foundation model scoring your data, from SQL. (On a held-out split
`mitra` gets iris **96%** right; see [`examples/`](../../examples/README.md).)

## Next

- Score a separate test table, choose features, options and output columns: [Functions](../api/01-functions.md)
- Compare models and pick one: [Choosing a model](02-choosing-a-model.md)
- Measure the predictions: [Evaluate and compare](03-evaluate-and-compare.md)
- Use a GPU: [GPU acceleration](04-gpu-acceleration.md)
