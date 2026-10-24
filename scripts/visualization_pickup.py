#!/usr/bin/env python3
"""Does anofox-visualization pick up anofox_tabfm's contract producers?

anofox-visualization plots any table that follows the anofox integration
contract (anofox-visualization docs/plans/integration-contract.md): its
`anofox_plot(tbl)` picks the chart from the columns, and producers advertise
themselves with `duckdb_functions()` tags
`{'anofox.contract': '1', 'anofox.output': '<schema>'}`. This script is the
definition of done for tabfm's producers (docs/ANOFOX_CONTRACT_PLAN.md). It
loads the tabfm artifact that was just built next to a released
anofox_visualization in a stock DuckDB CLI and checks, each in a fresh process:

  1. both extensions load in either order, and `anofox_contract_version()`
     (registered by whichever loads first) exists once and returns '1';
  2. discovery: the tagged tabfm producers are exactly EXPECTED_PRODUCERS, and
     the visualization has the plot macro for each one's schema (and, when its
     build carries `anofox.consumes` tags, declares that schema);
  3. every producer's output renders: `anofox_plot(t)` is an SVG with no
     render warning, and it is the same SVG as the schema's own macro, so the
     auto-detection chose that schema; every model (or class) is drawn. A
     matrix is one plot per model (the visualization asks for GROUP BY
     model_id), and so are multiclass curves (it keys curve lines by `series`
     or by `model_id`, not both); two-class curves compare models in one plot;
  4. the values survive: the heatmap's cells add up to the scored rows, and
     the ROC legend's AUC (the visualization integrates the curve) equals
     tabfm_roc_auc for that class;
  5. predictions scored by tabfm itself (the offline fixture model in
     test/fixtures, with two different contexts as two models) plot the same way.

    python3 scripts/visualization_pickup.py \\
        build/release/extension/anofox_tabfm/anofox_tabfm.duckdb_extension

The DuckDB CLI is downloaded (--duckdb-version, default v1.5.6) unless
--duckdb names one. The visualization comes from its project repository
(--viz-repo or VIZ_REPO; `community` for the community repository).
Telemetry is off. Exit status 1 when any check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_test import _download_duckdb_cli  # noqa: E402  # pyright: ignore[reportMissingImports]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The tagged producers this extension ships: function -> contract schema.
# Kept apart from the C++ list (src/tabfm_contract.cpp) on purpose, so a tag
# that goes missing or appears unplanned fails here.
EXPECTED_PRODUCERS = {
    name: schema
    for base, schema in (("classification_metrics_by", "summary"), ("regression_metrics_by", "summary"),
                         ("confusion_matrix_by", "matrix"), ("augment_by", "obs"),
                         ("roc_curve_by", "curve"), ("calibration_curve_by", "curve"))
    for name in (f"tabfm_{base}", f"anofox_tabfm_{base}")
}

# The visualization's macro for each schema; anofox_plot draws `obs` as the plot.lm diagnostics grid.
SCHEMA_MACRO = {"summary": "anofox_plot_summary", "matrix": "anofox_plot_matrix", "obs": "anofox_plot_diagnostics",
                "curve": "anofox_plot_curve"}

DEFAULT_VIZ_REPO = "https://datazoode.github.io/anofox-visualization/extensions"

# Two classifiers over three classes and two regressors (a NULL actual is a scored row).
FIXTURE = """
CREATE TABLE scored AS SELECT * FROM (VALUES
    ('a', 'x', 'x', MAP {'x': 0.7, 'y': 0.2, 'z': 0.1}), ('a', 'x', 'y', MAP {'x': 0.3, 'y': 0.5, 'z': 0.2}),
    ('a', 'y', 'y', MAP {'x': 0.1, 'y': 0.8, 'z': 0.1}), ('a', 'y', 'y', MAP {'x': 0.2, 'y': 0.6, 'z': 0.2}),
    ('a', 'z', 'z', MAP {'x': 0.1, 'y': 0.2, 'z': 0.7}), ('a', 'z', 'x', MAP {'x': 0.5, 'y': 0.1, 'z': 0.4}),
    ('b', 'x', 'x', MAP {'x': 0.6, 'y': 0.3, 'z': 0.1}), ('b', 'x', 'x', MAP {'x': 0.5, 'y': 0.4, 'z': 0.1}),
    ('b', 'y', 'z', MAP {'x': 0.2, 'y': 0.3, 'z': 0.5}), ('b', 'y', 'y', MAP {'x': 0.3, 'y': 0.4, 'z': 0.3}),
    ('b', 'z', 'z', MAP {'x': 0.2, 'y': 0.2, 'z': 0.6}), ('b', 'z', 'z', MAP {'x': 0.1, 'y': 0.1, 'z': 0.8})
) t(model, actual, yhat, proba);
CREATE TABLE two_class AS SELECT model, actual, MAP {'no': 1 - p, 'yes': p} AS proba,
    CASE WHEN p >= 0.5 THEN 'yes' ELSE 'no' END AS yhat FROM (VALUES
    ('a', 'yes', 0.9), ('a', 'yes', 0.7), ('a', 'no', 0.6), ('a', 'yes', 0.4), ('a', 'no', 0.2), ('a', 'no', 0.1),
    ('b', 'yes', 0.8), ('b', 'no', 0.8), ('b', 'yes', 0.55), ('b', 'no', 0.3), ('b', 'yes', 0.3), ('b', 'no', 0.05)
) t(model, actual, p);
CREATE TABLE fitted AS SELECT * FROM (VALUES
    ('r', 1, 1.0, 1.5), ('r', 2, 2.0, 1.5), ('r', 3, 3.0, 3.5), ('r', 4, 4.0, 3.0), ('r', 5, 0.0, 0.5),
    ('r', 6, NULL, 2.0), ('s', 1, 1.0, 1.2), ('s', 2, 2.0, 2.4), ('s', 3, 3.0, 2.7), ('s', 4, 4.0, 4.4),
    ('s', 5, 5.0, 4.6)
) t(model, id, actual, yhat);
"""

# Predictions scored by tabfm itself, offline: the fixture model with all labelled rows as context
# ('ctx-all') and with every other one ('ctx-half'), scoring the same held-out rows.
TABFM_FIXTURE = f"""
CALL tabfm_register_model(id := 'fixture', base_dir := '{REPO}/test/fixtures',
    classification_graph := 'graph_fixture.onnx', classification_weights := 'model.safetensors',
    classification_tensor_map := 'tensor_map_fixture.json', license := 'fixture-mit',
    preprocessing_profile := 'tabfm_v1_minimal');
CREATE TABLE rows AS SELECT i, i * 0.1 AS x, (i % 3) * 1.0 + (i % 7) * 0.05 AS z,
    'c' || (i % 3)::VARCHAR AS label FROM range(60) r(i);
CREATE TABLE ctx_all AS SELECT x, z, label FROM rows WHERE i < 40;
CREATE TABLE ctx_half AS SELECT x, z, label FROM rows WHERE i < 40 AND i % 2 = 0;
CREATE TABLE held AS SELECT x, z FROM rows WHERE i >= 40;
CREATE TABLE truth AS SELECT x, z, label FROM rows WHERE i >= 40;
CREATE TABLE tabfm_scored AS
SELECT 'ctx-all' AS model, truth.label AS actual, p.yhat, p.proba
FROM tabfm_classify('ctx_all', 'label', test := 'held', model := 'fixture') p JOIN truth USING (x, z)
UNION ALL
SELECT 'ctx-half', truth.label, p.yhat, p.proba
FROM tabfm_classify('ctx_half', 'label', test := 'held', model := 'fixture') p JOIN truth USING (x, z);
"""

CLASSIFICATION_CALLS = {
    "tabfm_classification_metrics_by": "tabfm_classification_metrics_by('{src}', 'model', 'actual', 'yhat', 'proba')",
    "tabfm_confusion_matrix_by": "tabfm_confusion_matrix_by('{src}', 'model', 'actual', 'yhat')",
    "tabfm_roc_curve_by": "tabfm_roc_curve_by('{src}', 'model', 'actual', 'proba')",
    "tabfm_calibration_curve_by": "tabfm_calibration_curve_by('{src}', 'model', 'actual', 'proba', n_bins := 5)",
}
REGRESSION_CALLS = {
    "tabfm_regression_metrics_by": "tabfm_regression_metrics_by('{src}', 'model', 'actual', 'yhat')",
    "tabfm_augment_by": "tabfm_augment_by('{src}', 'model', 'actual', 'yhat', row_id_col := 'id')",
}

# Both renders get the same size: anofox_plot and the schema macros have different default sizes.
SIZE = "width := 640, height := 400"


class Pickup:
    def __init__(self, duckdb: str, ext: str, viz_repo: str, home: str):
        self.duckdb, self.ext, self.viz_repo, self.home = duckdb, ext, viz_repo, home
        self.failures: list[str] = []

    def run(self, sql: str, load: str = "tabfm,viz") -> tuple[bool, str, list[dict]]:
        """`sql` in a fresh `duckdb -unsigned -json` process after the LOADs → (ok, output, last result rows)."""
        names = {"tabfm": "anofox_tabfm", "viz": "anofox_visualization"}
        loads = "".join(f"LOAD {names[x]};\n" for x in load.split(",") if x)
        env = dict(os.environ, HOME=self.home, USERPROFILE=self.home, DATAZOO_DISABLE_TELEMETRY="1")
        p = subprocess.run([self.duckdb, "-unsigned", "-json", "-c", loads + sql],
                           capture_output=True, text=True, env=env, timeout=600)
        stdout = p.stdout or ""
        rows: list[dict] = []
        dec, i = json.JSONDecoder(), 0
        while (j := stdout.find("[", i)) >= 0:
            try:
                parsed, i = dec.raw_decode(stdout, j)
                if isinstance(parsed, list):
                    rows = parsed
            except ValueError:
                i = j + 1
        return p.returncode == 0, stdout + (p.stderr or ""), rows

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"\n       {detail}" if detail and not ok else ""))
        if not ok:
            self.failures.append(name)
        return ok

    def install(self) -> None:
        src = "community" if self.viz_repo == "community" else f"'{self.viz_repo}'"
        ok, out, _ = self.run(f"INSTALL '{self.ext}';\nFORCE INSTALL anofox_visualization FROM {src};", load="")
        if not ok:
            raise SystemExit(f"Visualization pickup FAILED: could not install the extensions\n{out}")
        _, _, rows = self.run("SELECT extension_name, extension_version FROM duckdb_extensions() "
                              "WHERE extension_name IN ('anofox_tabfm', 'anofox_visualization') ORDER BY 1;")
        for r in rows:
            print(f"  {r['extension_name']} {r['extension_version']}")

    # ── 1 ──
    def load_order(self) -> None:
        print("[1] loading in either order, one shared anofox_contract_version()")
        for order in ("tabfm,viz", "viz,tabfm"):
            ok, out, rows = self.run("SELECT count(*) AS n, any_value(anofox_contract_version()) AS v "
                                     "FROM duckdb_functions() WHERE function_name = 'anofox_contract_version';",
                                     load=order)
            self.check(f"LOAD {order.replace(',', ' then ')}", ok and rows == [{"n": 1, "v": "1"}],
                       out.strip()[-500:])

    # ── 2 ──
    def discovery(self) -> None:
        print("[2] discovery through duckdb_functions() tags")
        ok, out, rows = self.run("SELECT function_name AS f, tags['anofox.output'] AS s FROM duckdb_functions() "
                                 "WHERE tags['anofox.contract'] = '1' AND (function_name LIKE 'tabfm\\_%' ESCAPE '\\' "
                                 "OR function_name LIKE 'anofox\\_tabfm\\_%' ESCAPE '\\');")
        found = {r["f"]: r["s"] for r in rows}
        self.check("tagged tabfm producers are exactly the expected ones", ok and found == EXPECTED_PRODUCERS,
                   f"expected {EXPECTED_PRODUCERS}\n       found    {found}\n       {out.strip()[-300:]}")
        ok, out, rows = self.run("SELECT function_name AS f, tags['anofox.consumes'] AS c FROM duckdb_functions() "
                                 "WHERE function_name LIKE 'anofox\\_plot%' ESCAPE '\\';")
        macros = {r["f"]: r["c"] for r in rows}
        consumes = macros.get("anofox_plot") or ""
        for schema in sorted(set(EXPECTED_PRODUCERS.values())):
            self.check(f"the visualization has {SCHEMA_MACRO[schema]} for '{schema}'",
                       ok and SCHEMA_MACRO[schema] in macros, out[-300:])
            if consumes:  # the community (C++) build tags its consumers; the C-API build has no tags
                self.check(f"anofox_plot declares it consumes '{schema}'", schema in consumes.split(","), consumes)

    # ── 3 / 4 / 5 ──
    def render_one(self, producer: str, call: str, setup: str, per_model: bool) -> None:
        """One plot of every model, or (per_model) one plot per model. The visualization keys curve lines
        by `series` when it varies, else by `model_id` (never both), so multiclass curves - one series per
        class - are drawn per model, like matrices (one matrix per plot)."""
        schema = EXPECTED_PRODUCERS[producer]
        macro = SCHEMA_MACRO[schema]
        per = "model_id" if per_model else "NULL"
        series = ("list(DISTINCT series ORDER BY series) FILTER (WHERE series IS NOT NULL)" if schema == "curve"
                  else "NULL::VARCHAR[]")
        ok, out, rows = self.run(
            f"{setup}\nCREATE TABLE t AS SELECT * FROM {call};\n"
            f"SELECT {per}::VARCHAR AS model, anofox_plot(t, {SIZE}) AS auto, {macro}(t, {SIZE}) AS specific, "
            f"list(DISTINCT model_id ORDER BY model_id) AS models, {series} AS series "
            f"FROM t GROUP BY ALL ORDER BY 1;")
        if not self.check(f"{producer}: anofox_plot renders", ok and len(rows) >= 1, out.strip()[-600:]):
            return
        for r in rows:
            auto, label = r["auto"] or "", producer + (f" [{r['model']}]" if r["model"] else "")
            self.check(f"{label}: an SVG", auto.startswith("<svg"), auto[:200])
            warn = re.search(r'data-warnings="([^"]*)"', auto)
            self.check(f"{label}: no render warnings", warn is None, warn.group(1) if warn else "")
            self.check(f"{label}: auto-detected as '{schema}' (same SVG as {macro})", auto == r["specific"])
            if schema == "curve":
                drawn = sorted({x.split(" · ")[0] for x in re.findall(r'data-series="([^"]*)"', auto)})
                want = sorted(r["series"] or []) if len(r["series"] or []) > 1 else r["models"]
                self.check(f"{label}: every {'class' if per_model else 'model'} drawn as a line", drawn == want,
                           f"want {want}, drawn {drawn}")
            elif schema != "matrix":
                self.check(f"{label}: every model drawn", all(f">{m}<" in auto for m in r["models"]),
                           str(r["models"]))
        if per_model:
            self.check(f"{producer}: one plot per model", len(rows) >= 2 and all(r["model"] for r in rows),
                       str([r["model"] for r in rows]))

    def roc_auc(self, setup: str, src: str, per_model: bool) -> None:
        """The legend's AUC (the visualization integrates the curve) against tabfm_roc_auc of the same class."""
        roc = CLASSIFICATION_CALLS["tabfm_roc_curve_by"].format(src=src)
        per = "model_id" if per_model else "NULL"
        key = "series" if per_model else "model_id"
        ok, out, rows = self.run(
            f"{setup}\nCREATE TABLE t AS SELECT * FROM {roc};\n"
            "CREATE TABLE want AS SELECT model_id, series, printf('%.2f', tabfm_roc_auc("
            "CASE WHEN s.actual = series THEN 'pos' ELSE 'neg' END, MAP {'pos': s.proba[series], "
            "'neg': 1 - s.proba[series]}, 'ovr')) AS auc "
            f"FROM (SELECT DISTINCT model_id, series FROM t) c JOIN {src} s ON s.model = c.model_id GROUP BY ALL;\n"
            f"SELECT {per}::VARCHAR AS model, anofox_plot(t) AS svg, (SELECT list(w.{key} || ' · AUC ' || w.auc "
            f"ORDER BY w.{key}) FROM want w WHERE {'w.model_id = t.model_id' if per_model else 'true'}) AS want "
            "FROM t GROUP BY ALL ORDER BY 1;")
        name = f"ROC legend AUC equals tabfm_roc_auc per {'class' if per_model else 'model'} ({src})"
        if not self.check(name, ok and len(rows) >= 1, out.strip()[-500:]):
            return
        for r in rows:
            series = sorted(set(re.findall(r'data-series="([^"]*)"', r["svg"])))
            self.check(f"{name}" + (f" [{r['model']}]" if r["model"] else ""), series == r["want"],
                       f"svg {series}, tabfm {r['want']}")

    def render(self, setup: str, label: str, calls: dict[str, str], src: str, multiclass: bool = False) -> None:
        print(f"[{label}] rendering every producer's output ({src})")
        for producer, call in calls.items():
            schema = EXPECTED_PRODUCERS[producer]
            self.render_one(producer, call.format(src=src), setup,
                            per_model=schema == "matrix" or (schema == "curve" and multiclass))

    def values(self, setup: str, src: str) -> None:
        cm = CLASSIFICATION_CALLS["tabfm_confusion_matrix_by"].format(src=src)
        ok, out, rows = self.run(
            f"{setup}\nCREATE TABLE t AS SELECT * FROM {cm};\n"
            f"SELECT model_id, anofox_plot(t) AS svg, (SELECT count(*) FROM {src} s WHERE s.model = t.model_id "
            f"AND actual IS NOT NULL AND yhat IS NOT NULL) AS n FROM t GROUP BY model_id ORDER BY 1;")
        if not self.check("heatmap cells add up to the scored rows (rendered)", ok and len(rows) >= 2,
                          out.strip()[-400:]):
            return
        for r in rows:
            cells = [float(v) for v in re.findall(r'data-value="([^"]*)"', r["svg"])]
            self.check(f"heatmap cells add up to the scored rows [{r['model_id']}]", sum(cells) == r["n"],
                       f"cells {cells}, rows {r['n']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="anofox-visualization picks up anofox_tabfm's contract producers")
    ap.add_argument("extension", help="the built anofox_tabfm.duckdb_extension")
    ap.add_argument("--duckdb", help="DuckDB CLI to use (default: download the official one)")
    ap.add_argument("--duckdb-version", default="v1.5.6")
    ap.add_argument("--arch", default="linux_amd64")
    ap.add_argument("--viz-repo", default=os.environ.get("VIZ_REPO", DEFAULT_VIZ_REPO),
                    help="extension repository of anofox_visualization ('community' for the community one)")
    args = ap.parse_args()

    ext = os.path.abspath(args.extension).replace("\\", "/")
    if not os.path.isfile(ext):
        print(f"Visualization pickup FAILED: artifact not found: {ext}")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        duckdb = args.duckdb or _download_duckdb_cli(args.duckdb_version, args.arch, tmp)
        home = os.path.join(tmp, "home")
        os.makedirs(home)
        print(f"\nVisualization pickup\n  tabfm  : {ext}\n  duckdb : {duckdb}\n  viz    : {args.viz_repo}")
        t = Pickup(duckdb, ext, args.viz_repo, home)
        t.install()
        t.load_order()
        t.discovery()
        t.render(FIXTURE, "3", CLASSIFICATION_CALLS, "scored", multiclass=True)
        t.render(FIXTURE, "3", CLASSIFICATION_CALLS, "two_class")
        t.render(FIXTURE, "3", REGRESSION_CALLS, "fitted")
        print("[4] the values survive")
        t.values(FIXTURE, "scored")
        t.roc_auc(FIXTURE, "scored", per_model=True)
        t.roc_auc(FIXTURE, "two_class", per_model=False)
        t.render(TABFM_FIXTURE, "5", CLASSIFICATION_CALLS, "tabfm_scored", multiclass=True)
        t.values(TABFM_FIXTURE, "tabfm_scored")
        t.roc_auc(TABFM_FIXTURE, "tabfm_scored", per_model=True)
    if t.failures:
        print(f"\nVisualization pickup FAILED ({len(t.failures)}):\n  " + "\n  ".join(t.failures))
        return 1
    print("\nVisualization pickup PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
