# Plotting with anofox-visualization

Six table macros turn scored rows into the long tables
[anofox-visualization](https://github.com/DataZooDE/anofox-visualization) plots
with `anofox_plot(t)`, one group per model (the anofox integration contract,
[`docs/ANOFOX_CONTRACT_PLAN.md`](../ANOFOX_CONTRACT_PLAN.md)). `data` is a
table name or a `'(SELECT …)'` string; the other arguments are column names.

| Macro | Gives | `anofox_plot` draws |
|---|---|---|
| `tabfm_classification_metrics_by(data, group_col, actual_col, predicted_col, proba_col)` | accuracy, ece, f1_macro, log_loss, roc_auc_ovr per group | model comparison |
| `tabfm_regression_metrics_by(data, group_col, actual_col, predicted_col)` | mae, mape, medae, r2, rmse per group | model comparison |
| `tabfm_confusion_matrix_by(data, group_col, actual_col, predicted_col)` | row (actual), col (predicted), value (count), zero cells included | a heatmap per model |
| `tabfm_augment_by(data, group_col, actual_col, predicted_col [, row_id_col])` | y, fitted, residual per row | the residual diagnostics grid |
| `tabfm_roc_curve_by(data, group_col, actual_col, proba_col [, pos_label, drop_intermediate])` | the ROC curve, one-vs-rest per class | ROC curves with their AUC |
| `tabfm_calibration_curve_by(data, group_col, actual_col, proba_col [, pos_label, n_bins])` | the reliability curve per class, 95% Wilson band | calibration curves |

Every macro ends with `row_order`, the contract's optional 1-based ordinal: the
order the rows come out in (metrics alphabetically, curves along the curve), so a
consumer that aggregates the rows can keep that order.

## Curves and classes

**Curves and classes.** A two-class table gives one curve per model, for the
greater label (scikit-learn's `classes_[1]`; the other curve is its mirror
image), so models compare in one plot. More classes give one curve per class
(`series` = the class); the mean area of a model's ROC curves is
`tabfm_roc_auc(avg := 'ovr')`. `pos_label := 'churned'` draws one class only.
The visualization keys curve lines by `series` or by `model_id`, not both, so
plot multiclass curves one model at a time (`GROUP BY model_id`), as with the
confusion matrix. The calibration curve bins each class's probability;
`tabfm_ece` bins the top-label confidence, so the curve does not reproduce the
ECE number.

The values are the metric aggregates above, so the plots and your own queries
agree. A metric that is undefined for a group (roc_auc_ovr with one class) is
left out.
