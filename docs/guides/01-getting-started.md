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

`SELECT * FROM tabfm_list_models();` shows the seven built-in models. `mitra` is a
good default — Apache-2.0, ~303 MB. It also has a permissive license. The extension ships only
**weight-free** graphs; the weights have to be downloaded from Hugging Face before using the model:

```sql
CALL tabfm_download('classification', model := 'mitra');    -- ~303 MB, cached in ~/.cache/anofox-tabfm
```

Models whose license needs an explicit nod (`tabfm-v1`, `tabpfn-v2`, `tabpfn-v2-5`,
`tabpfn-v3` — "(gated)" in the table below) additionally need
`SET anofox_tabfm_accept_hf_license = true;` before `tabfm_download`.

Some repositories are additionally **gated by Hugging Face** — accept the license on
the model page while signed in, then pass your token via a standard DuckDB secret:

```sql
CREATE SECRET hf (TYPE http, BEARER_TOKEN 'hf_xxx', SCOPE 'https://huggingface.co');
```

A download that needs this fails with an error naming both steps. See
[`docs/REAL_MODELS.md`](../REAL_MODELS.md#gated-huggingface-repositories).

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

Predict the hidden species — zero-shot, no training. Every function takes
`model :=` (or `SET anofox_tabfm_default_model = 'mitra'` once for the session):

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
