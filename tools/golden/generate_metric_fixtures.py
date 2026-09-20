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
Fixture: 3-row, 3-class classification

    actual    = ['cat', 'dog', 'cat']   (rows 1–3)
    predicted = ['cat', 'cat', 'cat']   (rows 1–3)

    Row 1: actual='cat', predicted='cat'  -> correct
    Row 2: actual='dog', predicted='cat'  -> wrong
    Row 3: actual='cat', predicted='cat'  -> correct

    accuracy = 2/3 = 0.666667 (rounded to 6 dp)
"""

import sys

try:
    from sklearn.metrics import accuracy_score
except ImportError:
    print("scikit-learn not found. Install with: pip install scikit-learn", file=sys.stderr)
    print("Golden values (pre-computed from sklearn 1.x):", file=sys.stderr)
    print("  accuracy_score(['cat','dog','cat'], ['cat','cat','cat']) = 0.6666666666666666", file=sys.stderr)
    sys.exit(1)

# 3-class fixture used in test/sql/tabfm_metrics_classification.test
actual    = ['cat', 'dog', 'cat']
predicted = ['cat', 'cat', 'cat']

acc = accuracy_score(actual, predicted)
print("=== tabfm_accuracy golden values ===")
print(f"fixture: actual={actual}, predicted={predicted}")
print(f"accuracy_score = {acc}")
print(f"round(accuracy_score, 6) = {round(acc, 6)}")
print()
print("Use in .test file:")
print("    query I")
print("    SELECT round(tabfm_accuracy(actual, predicted), 6) FROM preds")
print("    ----")
print(f"    {round(acc, 6)}")
print()

# NULL-skip fixture: 4 rows, row 2 has NULL actual, row 4 has NULL predicted
# Valid rows: row 1 (cat→cat, correct), row 3 (dog→dog, correct)
# = 2/2 = 1.0
actual_with_null    = ['cat', None, 'dog', 'cat']
predicted_with_null = ['cat', 'cat', 'dog', None]
valid_pairs = [(a, p) for a, p in zip(actual_with_null, predicted_with_null)
               if a is not None and p is not None]
acc_null_skip = accuracy_score([a for a, p in valid_pairs], [p for a, p in valid_pairs])
print("=== NULL-skip golden values ===")
print(f"fixture (with NULLs): actual={actual_with_null}, predicted={predicted_with_null}")
print(f"valid pairs: {valid_pairs}")
print(f"accuracy_score (valid only) = {acc_null_skip}")
print(f"round(accuracy_score_null_skip, 6) = {round(acc_null_skip, 6)}")
