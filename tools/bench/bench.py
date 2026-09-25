#!/usr/bin/env python3
"""Reproducible model x device x shape timings for anofox-tabfm.

Why this exists: every performance number in docs/ was produced by hand with
`.timer on` in a shell and pasted into a table. That makes the numbers
unreproducible and unauditable -- you cannot tell whether two rows were
measured on the same build, the same context size, or even the same device,
and a device that silently fell back to the CPU looks identical to one that
did not.

So the rule this harness enforces, from tools/gpu_test/scenarios/README.md:
EVERY measurement carries the device that actually served it, read from
tabfm_models() after the query. A row whose served_by is not what was asked
for is reported as such, never averaged in.

Usage:
    python3 tools/bench/bench.py --duckdb ./build/release/duckdb \\
        --models mitra,tabdpt --devices cpu,auto --rows 100,1000 --features 10

    # a device that needs a plugin
    python3 tools/bench/bench.py --devices rocm --ep-path ~/.cache/anofox-tabfm/runtime

Output is one JSON object per line (jsonl), so runs append and compare:
    python3 tools/bench/bench.py ... >> bench.jsonl
    duckdb -c "SELECT model, device, served_by, rows, median_ms FROM 'bench.jsonl'"
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time


def run_sql(duckdb_bin, sql, timeout, suppressions=None):
    """Run SQL through the DuckDB CLI, returning (stdout, stderr, ok)."""
    env = dict(os.environ)
    env.setdefault("DATAZOO_DISABLE_TELEMETRY", "1")
    # A sanitizer build otherwise fails every cell on ONNX Runtime's own
    # 14-byte static-init leak, which has nothing to do with what is being
    # measured (see test/lsan.supp).
    if suppressions and os.path.exists(suppressions):
        env["LSAN_OPTIONS"] = f"suppressions={suppressions}"
    try:
        proc = subprocess.run(
            [duckdb_bin, "-unsigned", "-init", "/dev/null", "-csv", "-noheader"],
            input=sql,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "", f"timed out after {timeout}s", False
    return proc.stdout.strip(), proc.stderr.strip(), proc.returncode == 0


def build_sql(model, device, ep_path, precision, n_rows, n_features, n_context, repeats, setup_sql=None):
    """One session: build the table, warm once, then time `repeats` runs.

    The warm-up is not optional and is excluded from the timings. A first call
    pays for session construction, weight injection and -- on MIGraphX -- a
    shape-bucket compile that can take minutes. Including it would report a
    one-off cost as the steady-state latency.
    """
    feature_cols = ", ".join(
        f"((i * {7 + k}) % 97)::DOUBLE AS f{k}" for k in range(n_features)
    )
    settings = [
        setup_sql or "",
        f"SET anofox_tabfm_device = '{device}';",
        f"SET anofox_tabfm_gpu_precision = '{precision}';",
        f"SET anofox_tabfm_max_rows = {max(n_rows, 10000)};",
    ]
    if ep_path:
        settings.append(f"SET anofox_tabfm_ep_path = '{ep_path}';")

    timed = "\n".join(
        f"SELECT count(*) FROM tabfm_classify('t', 'label', model := '{model}');"
        for _ in range(repeats)
    )
    return f"""
{chr(10).join(settings)}
CREATE TABLE t AS
SELECT {feature_cols},
       CASE WHEN i < {n_context} THEN ('c' || (i % 3)) ELSE NULL END AS label
FROM range({n_rows}) s(i);

-- warm: session build, weight injection, and any shape compile
SELECT count(*) FROM tabfm_classify('t', 'label', model := '{model}');
.print __WARM_DONE__
{timed}
.print __TIMED_DONE__
-- the only number that makes the rest trustworthy
SELECT 'SERVED_BY=' || coalesce(max(device), 'NONE')
FROM tabfm_models() WHERE loaded AND model = '{model}';
"""


def measure(args, model, device, n_rows, n_features):
    """Per-inference latency, by differencing two sessions.

    The CLI reports no per-statement timing in csv mode, so timing one session
    would fold in process startup, extension load, table build, session
    construction, weight injection and any shape compile -- on a 6.5GB model
    that dwarfs the inference being measured.

    Instead: run the identical session twice, once with N timed repeats and
    once with none. Everything except the N inferences is common to both and
    cancels. What remains, divided by N, is the warm per-inference cost.
    """
    n_context = max(1, int(n_rows * args.context_fraction))
    supp = os.path.join(os.path.dirname(__file__), "..", "..", "test", "lsan.supp")

    def session(repeats):
        sql = build_sql(model, device, args.ep_path, args.precision,
                        n_rows, n_features, n_context, repeats, args.setup_sql)
        start = time.perf_counter()
        out, err, ok = run_sql(args.duckdb, sql, args.timeout, supp)
        return (time.perf_counter() - start) * 1000.0, out, err, ok

    # Baseline first: if the cell is going to fail, fail before paying for N.
    base_ms, base_out, base_err, base_ok = session(0)
    ok = base_ok and "__TIMED_DONE__" in base_out
    full_ms = base_ms
    out, err = base_out, base_err
    if ok:
        full_ms, out, err, full_ok = session(args.repeats)
        ok = full_ok and "__TIMED_DONE__" in out

    served_by = "NONE"
    for line in out.splitlines():
        if line.startswith("SERVED_BY="):
            served_by = line.split("=", 1)[1]

    per_call = None
    if ok and args.repeats > 0:
        delta = full_ms - base_ms
        # A negative or trivial delta means the inference is lost in the noise
        # of process startup at this size; report it as unmeasurable rather
        # than as a suspiciously fast number.
        per_call = round(delta / args.repeats, 2) if delta > 0 else None

    record = {
        "model": model,
        "device_setting": device,
        "served_by": served_by,
        "rows": n_rows,
        "features": n_features,
        "context_rows": n_context,
        "repeats": args.repeats,
        "precision": args.precision,
        "ok": ok,
        "per_call_ms": per_call,
        "metric": "warm_per_inference_ms_by_session_differencing",
        "session_ms_with_repeats": round(full_ms, 1),
        "session_ms_baseline": round(base_ms, 1),
        "host": platform.node(),
        "arch": platform.machine(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not ok:
        record["error"] = (err or out)[:500]
    # The check that makes every other field meaningful: "cpu and gpu agree" is
    # also what a silent CPU fallback prints.
    if device != "auto" and served_by != "NONE":
        record["served_as_requested"] = (served_by.split(":")[0] == device)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duckdb", default="./build/release/duckdb", help="DuckDB binary with the extension built in")
    ap.add_argument("--models", default="mitra", help="comma-separated model ids")
    ap.add_argument("--devices", default="cpu", help="comma-separated anofox_tabfm_device values")
    ap.add_argument("--rows", default="100,1000", help="comma-separated total row counts")
    ap.add_argument("--features", default="10", help="comma-separated feature counts")
    ap.add_argument("--context-fraction", type=float, default=0.8, help="share of rows used as context")
    ap.add_argument("--repeats", type=int, default=3, help="timed runs per cell, after one warm-up")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--ep-path", default="", help="plugin directory, for GPU devices")
    ap.add_argument("--setup-sql", default="", help="SQL run before each session (e.g. CALL tabfm_register_model(...) "
                                                   "for a model that is not built in). Runs in BOTH the timed and "
                                                   "baseline sessions, so it cancels out of the differenced number.")
    ap.add_argument("--timeout", type=int, default=3600, help="per-cell seconds (MIGraphX cold compile is minutes)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    row_counts = [int(r) for r in args.rows.split(",")]
    feature_counts = [int(f) for f in args.features.split(",")]

    any_mismatch = False
    for model in models:
        for device in devices:
            for n_rows in row_counts:
                for n_features in feature_counts:
                    record = measure(args, model, device, n_rows, n_features)
                    print(json.dumps(record), flush=True)
                    if record.get("served_as_requested") is False:
                        any_mismatch = True
                        print(
                            f"WARNING: asked for '{device}' but {model} was served by "
                            f"'{record['served_by']}' — this row is not a measurement of {device}",
                            file=sys.stderr,
                        )
                    elif not record["ok"]:
                        print(f"FAILED: {model}/{device}/{n_rows}x{n_features}: "
                              f"{record.get('error', 'unknown')}", file=sys.stderr)
    # A silent fallback must not look like a clean run.
    return 2 if any_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
