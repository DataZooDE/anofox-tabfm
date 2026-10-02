#!/usr/bin/env python3
"""Fail if code decides "is this row context or query" by reading a target with a bare IsNull().

A target is MISSING when it is NULL *or* NaN, and +/-Infinity is an error; that rule lives in
ClassifyTargetValue (src/include/tabfm_preprocess.hpp). The bug this guards against was several places
each deciding for themselves with IsNull(), so a NaN-marked row was a query row in one place and a
training row in another. A source-level check cannot prove the rule is applied, but it catches the way
the bug came back last time: a new entry point reading the target the old way.

Flagged (the target is addressed by its index or column variable):
    <expr>[...target_idx].IsNull()          GetValue(target_col, ...).IsNull()
Silence a line that is genuinely about SQL NULL with a trailing  // target-null-ok: <reason>

usage: check_target_null_reads.py [files...]   (default: src/*.cpp)
"""
import pathlib
import re
import sys

PATTERNS = [
    re.compile(r"target_idx\s*\]\s*\.IsNull\s*\("),
    re.compile(r"GetValue\s*\(\s*target_col\b[^)]*\)\s*\.IsNull\s*\("),
]


def scan(path: pathlib.Path):
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if "target-null-ok" in line:
            continue
        if any(p.search(line) for p in PATTERNS):
            yield number, line.strip()


def main(argv):
    files = [pathlib.Path(a) for a in argv] or sorted(pathlib.Path("src").glob("*.cpp"))
    bad = 0
    for path in files:
        for number, line in scan(path):
            print(f"{path}:{number}: bare IsNull() on a target: {line}")
            bad += 1
    if bad:
        print(f"\n{bad} site(s). Use ClassifyTargetValue(...) == TargetValueKind::MISSING (NULL or NaN), or mark a "
              "genuine SQL-NULL test with '// target-null-ok: <reason>'.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
