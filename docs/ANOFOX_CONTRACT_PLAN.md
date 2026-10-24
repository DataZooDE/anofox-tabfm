# anofox integration contract: plan

Goal: tabfm results plot in anofox-visualization with `anofox_plot(t)` and no
hand-written SQL. The contract is anofox-visualization
`docs/plans/integration-contract.md` (v1). It names tabfm as a target. anofox-decide
joined it in DataZooDE/anofox-decide#37; this plan follows the same pattern.

## What the contract asks for

- Long tables in a known schema, detected from their column names.
- Producers tagged in `duckdb_functions()`:
  `{'anofox.output': '<schema>', 'anofox.contract': '1'}`.
- Every function of the extension tagged `anofox.family = 'tabfm'`.
- `anofox_contract_version()` returns `'1'`. It is registered only when no
  other anofox extension has registered it yet, so load order does not matter.

## Producers

All producers are table macros in tabfm's style: `data` is a table name or a
`'(SELECT …)'` string, and the column arguments are column names as strings.
The macros run through `query()` and quote identifiers with
`replace(col, '"', '""')`, as `tabfm_confusion_matrix` does. Every producer
keeps the group column under its own name and adds `model_id` (the group value
as VARCHAR). Every producer also exists as `anofox_tabfm_*`.

| Producer | Schema | Columns |
|---|---|---|
| `tabfm_roc_curve_by(data, group_col, actual_col, proba_col [, pos_label, drop_intermediate := true])` | `curve` (`roc`) | <group>, model_id, series, curve_type, x, y, y_low, y_high, threshold, weight, n_censor |
| `tabfm_calibration_curve_by(data, group_col, actual_col, proba_col [, pos_label, n_bins := 10])` | `curve` (`calibration`) | same; Wilson 95% in y_low / y_high, bin count in weight |
| `tabfm_classification_metrics_by(data, group_col, actual_col, predicted_col, proba_col)` | `summary` | <group>, model_id, metric, value, conf_low, conf_high |
| `tabfm_regression_metrics_by(data, group_col, actual_col, predicted_col)` | `summary` | same |
| `tabfm_confusion_matrix_by(data, group_col, actual_col, predicted_col)` | `matrix` | <group>, model_id, row (actual), col (predicted), value (count) |
| `tabfm_augment_by(data, group_col, actual_col, predicted_col [, row_id_col])` | `obs` | <group>, model_id, row_id, y, fitted, residual |

The curve columns are exactly anofox-statistics' `roc_curve_by` and
anofox-decide's `decide_roc_curve_by`, so curves from all three `UNION ALL`
into one plot.

### Decisions

- **Multiclass curves are one-vs-rest, one series per class.** `proba` is
  `MAP(label → probability)`. The ROC and calibration macros unnest it and draw
  one curve per class, with `series` = the class label. `tabfm_roc_auc(avg := 'ovr')`
  is the macro average of exactly these curves.
- **Binary data gives one curve, not two.** With two classes the two
  one-vs-rest curves are mirror images with the same AUC. The macros draw the
  greater label (scikit-learn's `classes_[1]`). `pos_label :=` picks the
  class explicitly and also works for multiclass data.
- **The calibration curve is per class, not top-label.** `tabfm_ece` bins the
  top-label confidence (Guo et al.). A reliability diagram per class needs
  x = mean P(class). README and the function description say the curve does
  not reproduce the ECE number (the same note decide has).
- **`tabfm_confusion_matrix` stays as it is.** Its `actual, predicted, count`
  columns are documented and in use, and the visualization does not read
  `actual` / `predicted` as row / col. The new `_by` macro is the tagged
  producer.
- **Metric names in `summary`:** classification gives `accuracy`, `f1_macro`,
  `log_loss`, `roc_auc_ovr`, `ece`; regression gives `rmse`, `mae`, `medae`,
  `mape`. A metric that is undefined for a group (one class, a zero actual
  for MAPE) is left out, not written as NULL or NaN.
- **Pure SQL over the existing aggregates.** No new C++ aggregates, so one
  macro text builds on DuckDB 1.5.6 and 2.0.
- **Errors name the function and the remedy:** a probability outside [0, 1],
  a non-numeric regression target, `n_bins < 1`. A `proba` column that is not
  a MAP fails when DuckDB binds the query, before a macro can check a value:
  the macros cast it to `MAP(VARCHAR, DOUBLE)`, so DuckDB's cast error names
  the type that is needed.

## Not in scope

- `tabfm_cross_validate` output (fold_id, metric_value, metric_std) is not
  re-shaped. It can later produce `summary` rows per fold.
- A `prediction` producer for regression intervals needs a distribution output
  from `tabfm_regress`, which does not exist yet (README, "Proper scoring rules").
- A `distribution` producer for `proba` (one row per option): small, but it
  belongs with the visualization's decide distribution work.

## Definition of done: anofox-visualization picks the producers up

`scripts/visualization_pickup.py`, ported from anofox-decide, plus
`make test_visualization` and a CI job **Visualization pickup** that deploy
`needs:`. It runs the built artifact next to a released anofox_visualization in
the official DuckDB v1.5.6 CLI and checks:

1. both load orders work, and `anofox_contract_version()` exists once and returns `'1'`;
2. the tagged tabfm producers are exactly the expected twelve, and the
   visualization has the schema macros (`anofox_plot_curve`, `_summary`,
   `_matrix`, `_obs`);
3. for every producer, `anofox_plot(t)` is an SVG with no `data-warnings` and is
   the same SVG as the schema's own macro (at the same size: the defaults
   differ), and every model is drawn;
4. the ROC legend's AUC per series equals the one-vs-rest AUC from tabfm's own
   metric;
5. probabilities scored by tabfm itself (the offline fixture model in
   `test/fixtures`, as two "models") plot the same way.

What the visualization (v2026.10.24) does with the new schemas, found while
building the check:

- `matrix`: `anofox_plot` draws one matrix per plot. A table with several
  `model_id`s is an error that asks for `GROUP BY model_id`, so the check and
  the README plot one model at a time.
- `obs`: `anofox_plot` draws it as the plot.lm diagnostics grid, the same SVG as
  `anofox_plot_diagnostics` (not `anofox_plot_obs`'s default kind).

## Tests (red first)

`test/sql/tabfm_anofox_contract.test`:
- column names and types for every producer;
- ROC and calibration points against scikit-learn (binary, multiclass,
  `pos_label`, ties, `drop_intermediate := false`);
- metric values against scikit-learn and against tabfm's own aggregates;
- the confusion matrix and augment values on a small fixture;
- one-class groups, data given as a subquery string, quoted identifiers;
- the errors above;
- the tags: exactly the twelve producers, the family tag on every tabfm
  function and on nothing of DuckDB's.

`tabfm_function_docs.test` counts go up by the new functions.

## Delivery

1. PR 1: contract plumbing (version function, tags), `confusion_matrix_by`,
   both `metrics_by`, `augment_by`, the pickup script and CI job.
2. PR 2: `roc_curve_by` and `calibration_curve_by` (the multiclass part).
3. Release, then the community-extensions descriptor bump.
4. Tell the visualization side (anofox-visualization#172) the producer names,
   so its conformance harness gets recipes for them.
