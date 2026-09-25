#!/usr/bin/env python3
"""
tools/golden/generate_metric_fixtures.py — fixture-generation aid for metric tests.

Prints sklearn golden values for tabfm metric aggregates so committed hard-coded
values in test/sql/tabfm_metrics_classification.test (and the other .test files)
are traceable to a scikit-learn reference run.

Usage:
    uv run tools/golden/generate_metric_fixtures.py
    # or: python3 tools/golden/generate_metric_fixtures.py (requires scikit-learn)

NOTE: This script is a fixture-generation AID, not a build artifact. It is NOT
listed in CMakeLists.txt and is not run during CI. The committed hard-coded values
in the .test files are the source of truth; this script documents how they were
derived from sklearn.

Golden values used in test/sql/tabfm_metrics_classification.test
-------------------------------------------------------------------

CMET-01: tabfm_accuracy
  Fixture: 3-row, 3-class classification
    actual    = ['cat', 'dog', 'cat']
    predicted = ['cat', 'cat', 'cat']
    accuracy  = 2/3 = 0.666667

CMET-02: tabfm_f1, tabfm_precision, tabfm_recall
  Same 3-row fixture.
  F1 macro    = 0.400000
  F1 weighted = 0.533333
  F1 micro    = 0.666667
  Precision macro    = 0.333333
  Precision weighted = 0.444444
  Precision micro    = 0.666667
  Recall macro    = 0.500000
  Recall weighted = 0.666667
  Recall micro    = 0.666667

CMET-03: tabfm_log_loss
  3-row fixture with proba MAP(VARCHAR, DOUBLE):
    Row 1: actual=cat, proba={cat:0.7, dog:0.2, fish:0.1}
    Row 2: actual=dog, proba={cat:0.1, dog:0.8, fish:0.1}
    Row 3: actual=cat, proba={cat:0.5, dog:0.3, fish:0.2}
  log_loss = 0.424322 (eps=1e-15)

CMET-04: tabfm_roc_auc
  6-row, 3-class fixture WITH tied scores:
    actual = ['cat','cat','dog','dog','fish','fish']
    proba[cat]  = [0.8, 0.6, 0.6, 0.2, 0.1, 0.1]  <- tie at 0.6
    proba[dog]  = [0.1, 0.3, 0.3, 0.7, 0.1, 0.2]
    proba[fish] = [0.1, 0.1, 0.1, 0.1, 0.8, 0.7]
  ROC-AUC OvR = 0.958333
  ROC-AUC OvO = 0.958333

CMET-05: tabfm_confusion_matrix
  3-row fixture (same as accuracy):
    (cat,cat): count=2
    (dog,cat): count=1

CMET-06: tabfm_ece
  3-row proba fixture (same as log-loss):
    All argmax correct -> ECE = 0.333333
Golden values used in test/sql/tabfm_metrics_regression.test
-------------------------------------------------------------------

RMET-01: tabfm_rmse
  Fixture: 5-row regression
    actual    = [1.0, 2.0, 3.0, 4.0, 5.0]
    predicted = [1.1, 2.2, 2.9, 3.8, 5.5]
    residuals = [0.1, 0.2, 0.1, 0.2, 0.5]
    rmse      = sqrt((0.01+0.04+0.01+0.04+0.25)/5) = 0.264575

RMET-02: tabfm_mae
  Same 5-row fixture.
  mae = (0.1+0.2+0.1+0.2+0.5)/5 = 0.22

RMET-03: tabfm_r2
  Same 5-row fixture.
  r2 = 1 - SS_res/SS_tot = 0.965
  Constant-target edge cases (sklearn convention):
    - all actual=5.0, predicted=5.0 → 1.0
    - all actual=5.0, predicted imperfect → 0.0

RMET-04: tabfm_mape, tabfm_medae
  Same 5-row fixture (no zero actuals).
  MAPE (dimensionless ratio, not %) = 0.076667
  Zero-actual skip fixture: actual=[0.0,1.0,2.0], predicted=[0.5,1.5,2.5]
    Non-zero rows: (1.0,1.5),(2.0,2.5) → MAPE = (0.5 + 0.25)/2 = 0.375
  MedAE N=5: sorted residuals=[0.1,0.1,0.2,0.2,0.5] → median=0.2
  MedAE N=4: actual=[1.0,2.0,3.0,4.0], predicted=[1.1,2.2,2.9,3.8]
    sorted residuals=[0.1,0.1,0.2,0.2] → median=(0.1+0.2)/2=0.15
"""

import math
import sys

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        log_loss,
        roc_auc_score,
    )
except ImportError:
    print("scikit-learn not found. Install with: pip install scikit-learn", file=sys.stderr)
    print("\nPre-computed golden values (from sklearn 1.x):")
    print("  accuracy     = 0.666667")
    print("  f1 macro     = 0.4")
    print("  f1 weighted  = 0.533333")
    print("  f1 micro     = 0.666667")
    print("  precision macro     = 0.333333")
    print("  precision weighted  = 0.444444")
    print("  precision micro     = 0.666667")
    print("  recall macro     = 0.5")
    print("  recall weighted  = 0.666667")
    print("  recall micro     = 0.666667")
    print("  log_loss     = 0.424322")
    print("  roc_auc ovr  = 0.958333")
    print("  roc_auc ovo  = 0.958333")
    print("  ece          = 0.333333")
    sys.exit(1)

# ============================================================
# CMET-01: accuracy fixture
# ============================================================
actual    = ['cat', 'dog', 'cat']
predicted = ['cat', 'cat', 'cat']

acc = accuracy_score(actual, predicted)
print("=== CMET-01: tabfm_accuracy ===")
print(f"fixture: actual={actual}, predicted={predicted}")
print(f"accuracy_score = {acc}")
print(f"round6 = {round(acc, 6)}")
print()

# ============================================================
# CMET-02: F1/precision/recall — same 3-row fixture
# ============================================================
print("=== CMET-02: tabfm_f1 / tabfm_precision / tabfm_recall ===")
print(f"fixture: actual={actual}, predicted={predicted}")
print()

for avg in ['macro', 'weighted', 'micro']:
    f1  = f1_score(actual, predicted, average=avg, zero_division=0)
    p   = precision_score(actual, predicted, average=avg, zero_division=0)
    r   = recall_score(actual, predicted, average=avg, zero_division=0)
    print(f"  avg={avg}:  f1={round(f1,6)}  precision={round(p,6)}  recall={round(r,6)}")

print()

# ============================================================
# CMET-03: log-loss — 3-row proba MAP fixture
# ============================================================
print("=== CMET-03: tabfm_log_loss ===")
actual_ll    = ['cat', 'dog', 'cat']
proba_matrix = [
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.5, 0.3, 0.2],
]
ll = log_loss(actual_ll, proba_matrix, labels=['cat', 'dog', 'fish'])
print(f"log_loss (eps=1e-15 default) = {ll}")
print(f"round6 = {round(ll, 6)}")
print()

# Manual clipping verification
eps = 1e-15
p_zero_clipped = max(eps, min(1.0 - eps, 0.0))
print(f"Clip test: p=0.0 -> {p_zero_clipped}, finite log-loss = {-math.log(p_zero_clipped):.6f}")
print()

# ============================================================
# CMET-04: ROC-AUC — 6-row 3-class tied-score fixture
# ============================================================
print("=== CMET-04: tabfm_roc_auc ===")
actual_auc = ['cat', 'cat', 'dog', 'dog', 'fish', 'fish']
proba_auc  = [
    {'cat': 0.8, 'dog': 0.1, 'fish': 0.1},
    {'cat': 0.6, 'dog': 0.3, 'fish': 0.1},
    {'cat': 0.6, 'dog': 0.3, 'fish': 0.1},   # tie in cat score
    {'cat': 0.2, 'dog': 0.7, 'fish': 0.1},
    {'cat': 0.1, 'dog': 0.1, 'fish': 0.8},
    {'cat': 0.1, 'dog': 0.2, 'fish': 0.7},
]
proba_matrix_auc = [[p['cat'], p['dog'], p['fish']] for p in proba_auc]
labels = ['cat', 'dog', 'fish']

ovr = roc_auc_score(actual_auc, proba_matrix_auc, multi_class='ovr',
                    labels=labels, average='macro')
ovo = roc_auc_score(actual_auc, proba_matrix_auc, multi_class='ovo',
                    labels=labels, average='macro')
print(f"ROC-AUC OvR macro = {ovr}  round6={round(ovr,6)}")
print(f"ROC-AUC OvO macro = {ovo}  round6={round(ovo,6)}")
print()

# ============================================================
# CMET-05: tabfm_confusion_matrix — same 3-row fixture
# ============================================================
print("=== CMET-05: tabfm_confusion_matrix ===")
print(f"fixture: actual={actual}, predicted={predicted}")
from collections import Counter
cm = Counter(zip(actual, predicted))
for (a, p), count in sorted(cm.items()):
    print(f"  (actual={a}, predicted={p}): count={count}")
print()

# ============================================================
# CMET-06: tabfm_ece — same 3-row proba MAP fixture
# ============================================================
print("=== CMET-06: tabfm_ece ===")
actual_ece    = ['cat', 'dog', 'cat']
proba_rows_ece = [
    {'cat': 0.7, 'dog': 0.2, 'fish': 0.1},
    {'cat': 0.1, 'dog': 0.8, 'fish': 0.1},
    {'cat': 0.5, 'dog': 0.3, 'fish': 0.2},
]

B = 10
bin_n      = [0]*B
bin_correct= [0]*B
bin_conf   = [0.0]*B
for a, pr in zip(actual_ece, proba_rows_ece):
    max_class = max(pr, key=pr.get)
    max_prob  = pr[max_class]
    b = min(int(max_prob * B), B-1)
    bin_n[b] += 1
    if max_class == a:
        bin_correct[b] += 1
    bin_conf[b] += max_prob

N = len(actual_ece)
ece = 0.0
for m in range(B):
    if bin_n[m] == 0:
        continue
    acc_m  = bin_correct[m] / bin_n[m]
    conf_m = bin_conf[m] / bin_n[m]
    ece   += (bin_n[m] / N) * abs(acc_m - conf_m)
print(f"ECE (10-bin) = {ece}  round6={round(ece,6)}")
