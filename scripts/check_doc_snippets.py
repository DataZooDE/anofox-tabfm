#!/usr/bin/env python3
"""scripts/check_doc_snippets.py -- keeps the user docs and the Claude Code skills honest.

Ported from anofox-forecast's checker, with three additions tabfm needs: a model
catalogue check, a settings check, and a mode that runs against a CI artifact so
the check can gate pull requests instead of being a manual release step.

Usage:
    python3 scripts/check_doc_snippets.py                 # full check, build/release/duckdb
    python3 scripts/check_doc_snippets.py FILE ...        # only these markdown files
    python3 scripts/check_doc_snippets.py --changed-since origin/main
    python3 scripts/check_doc_snippets.py --static        # no binary: manifests, versions, models
    python3 scripts/check_doc_snippets.py --extension anofox_tabfm.duckdb_extension \\
        [--duckdb /path/to/duckdb | --duckdb-version v1.5.6]   # CI: a packaged artifact

Scanned by default: README.md, docs/README.md, docs/api/*.md, docs/guides/*.md and every
plugins/anofox-tabfm/skills/**/SKILL.md. The design notes and plan history under docs/ are
not scanned: they describe code as it was.

Static checks (always run, need no binary):
  * the plugin manifests parse, agree on the plugin name, and point at real directories;
  * every SKILL.md has name (= its directory), description, version and user-invocable in
    its frontmatter, and its version (frontmatter and the "**Extension:**" header line)
    equals plugin.json's;
  * every built-in model (parsed from src/tabfm_registry.cpp) has a row in each model
    table listed in MODEL_TABLE_FILES, no row names an unknown model, and the README's
    "### N models" heading says N;
  * every model named as `model := '<id>'` or `default_model = '<id>'` exists.

Binary checks (skipped with --static):
  * the registry parse above equals tabfm_list_models();
  * every `tabfm_*` / `anofox_tabfm_*` identifier in a code span, in call syntax or in a
    SKILL.md description is a registered function or setting (or listed in
    scripts/doc_snippets/known_names.txt as a deliberate non-function name);
  * every ```sql fence runs. Each block runs in :memory:, in a fresh temporary HOME/cwd,
    after scripts/doc_snippets/fixtures.sql (the tables the docs use). A built-in model the
    block names in quotes ('mitra') is registered for that block as a stand-in for the
    committed, weight-free TabPFN fixture (test/fixtures/tabpfn, copied into the temp dir):
    the block's SQL is checked for real -- signatures, named parameters, option keys,
    output columns -- with no weights and no network. Classify-only models are registered
    classify-only, so an error example about them still errors.

Block statuses: PASS, FAIL, EXPECTED-ERROR (keyed in expected_errors.txt -- it must still
fail), SYNOPSIS (keyed in synopsis.txt, a signature or placeholder, never run), SKIP-NETWORK
(INSTALL, a URL, hf://, CREATE SECRET, tabfm_download / _runtime / _accelerate, or another
extension), SKIP-VERSION (first line `-- DuckDB 2.0+` on an older binary). Keys are
"<file><TAB><first non-comment line>". On a full scan, a support-file entry that matches
nothing is STALE.

Writes a TSV (status, file, line, first SQL line, detail) to --out (default
build/doc-snippets.tsv) and prints a one-line summary. Exit status: 0 clean, 1 on any FAIL
or STALE, 2 when the binary is missing or cannot be queried.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

SUPPORT_DIR = os.path.join(SCRIPT_DIR, "doc_snippets")
FIXTURES_FILE = os.path.join(SUPPORT_DIR, "fixtures.sql")
EXPECTED_ERRORS_FILE = os.path.join(SUPPORT_DIR, "expected_errors.txt")
SYNOPSIS_FILE = os.path.join(SUPPORT_DIR, "synopsis.txt")
REMOVED_NAMES_FILE = os.path.join(SUPPORT_DIR, "removed_names.txt")
KNOWN_NAMES_FILE = os.path.join(SUPPORT_DIR, "known_names.txt")
DEFAULT_DUCKDB = os.path.join(REPO_ROOT, "build", "release", "duckdb")
DEFAULT_OUT = os.path.join(REPO_ROOT, "build", "doc-snippets.tsv")
REGISTRY_SRC = os.path.join(REPO_ROOT, "src", "tabfm_registry.cpp")
FIXTURE_MODEL_DIR = os.path.join(REPO_ROOT, "test", "fixtures", "tabpfn")

PLUGIN_NAME = "anofox-tabfm"
MARKETPLACE_JSON = os.path.join(REPO_ROOT, ".claude-plugin", "marketplace.json")
PLUGIN_DIR = os.path.join(REPO_ROOT, "plugins", PLUGIN_NAME)
PLUGIN_JSON = os.path.join(PLUGIN_DIR, ".claude-plugin", "plugin.json")
SKILLS_DIR = os.path.join(PLUGIN_DIR, "skills")

SCAN_FILES = ["README.md", os.path.join("docs", "README.md")]
SCAN_DIRS = [os.path.join("docs", "api"), os.path.join("docs", "guides"),
             os.path.join("plugins", PLUGIN_NAME, "skills")]

# Files whose model table must list every built-in model, one row each.
MODEL_TABLE_FILES = [
    "README.md",
    os.path.join("docs", "api", "02-models-and-weights.md"),
    os.path.join("plugins", PLUGIN_NAME, "skills", "anofox-tabfm-models", "SKILL.md"),
]
# tabfm-v1 is the one built-in assembled in C++ from the manifests rather than a JSON entry.
REGISTRY_CPP_MODELS = {"tabfm-v1"}
# Model ids the docs use as placeholders or for a user's own registration.
PLACEHOLDER_MODELS = {"my", "fixture", "<id>", "<model>"}
CLASSIFY_ONLY = {"orion-bix", "orion-msp"}
REGRESS_ONLY = {"nori-30m"}

FENCE_RE = re.compile(r"^```[ \t]*([A-Za-z0-9_+-]*)[ \t]*$")
SETUP_RE = re.compile(
    r"\b(CREATE\s+(OR\s+REPLACE\s+)?(TEMP\s+|TEMPORARY\s+)?(TABLE|VIEW|MACRO)|INSERT)\b", re.IGNORECASE)
CREATE_NAME_RE = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP\s+|TEMPORARY\s+)?(?:TABLE|VIEW|MACRO)\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
INSERT_NAME_RE = re.compile(r"\bINSERT\s+INTO\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
PLACEHOLDER_ELLIPSIS_RE = re.compile(r"(?<!\d)(\.\.\.|…)(?!\d)")
NETWORK_RE = re.compile(
    r"\bINSTALL\b|https?://|s3://|hf://|\bCREATE\s+SECRET\b|\btabfm_download(_runtime)?\s*\(|"
    r"\btabfm_accelerate\s*\(|\banofox_visualization\b|\banofox_plot\s*\(",
    re.IGNORECASE)
V2_ONLY_RE = re.compile(r"\A\s*--\s*DuckDB 2\.0\+")

# An identifier, not a file name (tabfm_registry.cpp) or a path component (anofox_tabfm/...).
NAME_PAT = r"((?:anofox_)?tabfm_[a-z0-9_]*[a-z0-9])(?![a-z0-9_.*/{-])"
IDENT_RE = re.compile(r"(?<![A-Za-z0-9_])" + NAME_PAT)
BACKTICK_SPAN_RE = re.compile(r"`([^`\n]+)`")
CALL_IDENT_RE = re.compile(r"(?<![A-Za-z0-9_.])((?:anofox_)?tabfm_[a-z0-9_]+)\s*\(")
MODEL_ARG_RE = re.compile(r"\b(?:model\s*:=\s*|default_model\s*=\s*)'([^']*)'")
MODEL_ROW_RE = re.compile(r"^\|\s*`([a-z0-9][a-z0-9.-]*)`\s*\|")
MODELS_HEADING_RE = re.compile(r"^###\s+(\d+)\s+models\b", re.IGNORECASE)
HEADER_VERSION_RE = re.compile(r"\*\*Extension:\*\*\s*anofox_tabfm\s+v?(\d{4}\.\d{2}\.\d{2})")


# ---------------------------------------------------------------------------- helpers

def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_key_set(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f if line.strip() and not line.startswith("#")}


def load_name_list(path):
    """name<TAB>reason per line; blank and '#' lines ignored."""
    names = {}
    if not os.path.exists(path):
        return names
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            name, _, reason = line.partition("\t")
            names[name.strip()] = reason.strip()
    return names


def default_scan_files():
    files = [f for f in SCAN_FILES if os.path.exists(os.path.join(REPO_ROOT, f))]
    for scan_dir in SCAN_DIRS:
        abs_dir = os.path.join(REPO_ROOT, scan_dir)
        for root, dirs, fnames in os.walk(abs_dir):
            dirs.sort()
            for fn in sorted(fnames):
                if fn.endswith(".md"):
                    files.append(os.path.relpath(os.path.join(root, fn), REPO_ROOT))
    return files


def changed_md_files(ref):
    out = subprocess.run(["git", "diff", "--name-only", ref, "--"] + SCAN_FILES + SCAN_DIRS,
                         cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in (l.strip() for l in out.splitlines())
            if f.endswith(".md") and os.path.exists(os.path.join(REPO_ROOT, f))]


def extract_sql_blocks(text):
    """(block_text, start_line_idx) for every ```sql fence."""
    lines = text.splitlines()
    blocks, i = [], 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i].strip())
        if m and m.group(1).lower() == "sql":
            start, body = i, []
            i += 1
            while i < len(lines) and lines[i].strip() != "```":
                body.append(lines[i])
                i += 1
            blocks.append(("\n".join(body), start))
        i += 1
    return blocks


def first_non_comment_line(block_text):
    for line in block_text.splitlines():
        s = line.strip()
        if s and not s.startswith("--"):
            return s
    return next((l.strip() for l in block_text.splitlines() if l.strip()), "")


def strip_sql_comments(text):
    return "\n".join(line.split("--", 1)[0] for line in text.splitlines())


def setup_created_name(block_text):
    m = CREATE_NAME_RE.search(block_text) or INSERT_NAME_RE.search(block_text)
    return m.group(1) if m else None


def select_required_setup(target_text, setup_accum, fixture_names):
    """Setup blocks (doc order) the target references, transitively; names the target
    creates itself or fixtures.sql already provides are not pulled in."""
    shadowed = set(CREATE_NAME_RE.findall(strip_sql_comments(target_text))) | set(fixture_names)
    needed = [strip_sql_comments(target_text)]
    included, included_names = [], set()
    changed = True
    while changed:
        changed = False
        combined = "\n".join(needed)
        for name, block_text in setup_accum:
            if name in included_names or name in shadowed:
                continue
            if re.search(r"\b" + re.escape(name) + r"\b", combined):
                included.append(block_text)
                included_names.add(name)
                needed.append(strip_sql_comments(block_text))
                changed = True
    order = {id(bt): i for i, (_, bt) in enumerate(setup_accum)}
    included.sort(key=lambda bt: order.get(id(bt), 0))
    return included


def first_error_line(stderr, stdout):
    for stream in (stderr, stdout):
        for line in stream.splitlines():
            if line.strip() and "Error" in line:
                return line.strip()[:240]
    for stream in (stderr, stdout):
        for line in stream.splitlines():
            if line.strip():
                return line.strip()[:240]
    return ""


def sql_str(s):
    return "'" + s.replace("'", "''") + "'"


# ---------------------------------------------------------------------------- the binary

class Runner:
    def __init__(self, duckdb_bin, extension):
        self.duckdb = duckdb_bin
        self.load = f"LOAD {sql_str(extension)};" if extension else "LOAD anofox_tabfm;"

    def run(self, sql_text, models=(), timeout=180):
        """Run sql_text in :memory: with a fresh HOME/cwd. `models` are built-in ids to stand
        in for with the weight-free fixture for this block."""
        tmp = tempfile.mkdtemp(prefix="doc_snippets_")
        try:
            prelude = ["SET autoinstall_known_extensions = false;", "SET autoload_known_extensions = false;",
                       self.load, f"SET anofox_tabfm_cache_dir = {sql_str(os.path.join(tmp, 'cache'))};"]
            if models:
                fx = os.path.join(tmp, "fx")
                shutil.copytree(FIXTURE_MODEL_DIR, fx)
                for m in sorted(models):
                    args = [f"id := {sql_str(m)}", f"base_dir := {sql_str(fx)}",
                            "license := 'apache-2.0'", "commercial := true",
                            "preprocessing_profile := 'tabpfn_v2_raw'",
                            "max_rows := 4096", "max_features := 64", "max_classes := 4"]
                    if m not in REGRESS_ONLY:
                        args += ["classification_graph := 'graph_tabpfn_classification.onnx'",
                                 "classification_weights := 'model.safetensors'",
                                 "classification_tensor_map := 'tensor_map_tabpfn_classification.json'"]
                    if m not in CLASSIFY_ONLY:
                        args += ["regression_graph := 'graph_tabpfn_regression.onnx'",
                                 "regression_weights := 'model_regression.safetensors'",
                                 "regression_tensor_map := 'tensor_map_tabpfn_regression.json'"]
                    prelude.append(f"CALL tabfm_register_model({', '.join(args)});")
            env = dict(os.environ, HOME=tmp, DATAZOO_DISABLE_TELEMETRY="1", DATAZOO_NO_BANNER="1")
            try:
                r = subprocess.run([self.duckdb, "-unsigned", "-bail", "-list", ":memory:"],
                                   input="\n".join(prelude) + "\n" + sql_text, cwd=tmp,
                                   capture_output=True, text=True, timeout=timeout, env=env)
                return r.returncode, r.stdout, r.stderr
            except subprocess.TimeoutExpired:
                return 124, "", f"TIMEOUT after {timeout}s"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def query_set(self, sql):
        rc, out, err = self.run(sql)
        if rc != 0:
            print(f"ERROR: query failed: {sql}\n{(err.strip() or out.strip())[:500]}", file=sys.stderr)
            sys.exit(2)
        return {l.strip() for l in out.splitlines()[1:] if l.strip()}

    def duckdb_major(self):
        rc, out, _ = self.run("SELECT version();")
        m = re.search(r"v(\d+)\.", out or "")
        return int(m.group(1)) if rc == 0 and m else 0


# ---------------------------------------------------------------------------- static checks

class Report:
    def __init__(self):
        self.rows = []
        self.counts = {}

    def add(self, status, file, line="", first="", detail=""):
        self.rows.append((status, file, str(line), first, detail))
        self.counts[status] = self.counts.get(status, 0) + 1

    def fail(self, file, line, detail):
        self.add("FAIL", file, line, "", detail)


def registry_models_from_source():
    text = read_text(REGISTRY_SRC)
    return set(re.findall(r'"schema_version":\s*\d+,\s*"id":\s*"([a-z0-9.-]+)"', text)) | REGISTRY_CPP_MODELS


def parse_frontmatter(text):
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, 0
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fm, key = {}, None
            for line in lines[1:i]:
                m = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
                if m:
                    key, val = m.group(1), m.group(2).strip()
                    fm[key] = "" if val in (">", "|", ">-", "|-") else val
                elif key:
                    fm[key] = (fm[key] + " " + line.strip()).strip()
            return fm, i
    return None, 0


def check_plugin(report):
    rel = lambda p: os.path.relpath(p, REPO_ROOT)
    try:
        market = json.loads(read_text(MARKETPLACE_JSON))
        plugin = json.loads(read_text(PLUGIN_JSON))
    except (OSError, ValueError) as e:
        report.fail(".claude-plugin", "", f"plugin manifest missing or invalid: {e}")
        return None
    entries = [p for p in market.get("plugins", []) if p.get("name") == PLUGIN_NAME]
    if not entries:
        report.fail(rel(MARKETPLACE_JSON), "", f"no plugin named {PLUGIN_NAME}")
    for p in entries:
        src = os.path.normpath(os.path.join(REPO_ROOT, p.get("source", "")))
        if src != os.path.normpath(PLUGIN_DIR):
            report.fail(rel(MARKETPLACE_JSON), "", f"plugin source {p.get('source')} is not plugins/{PLUGIN_NAME}")
    if plugin.get("name") != PLUGIN_NAME:
        report.fail(rel(PLUGIN_JSON), "", f"plugin name {plugin.get('name')!r} != {PLUGIN_NAME}")
    version = plugin.get("version", "")
    if not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", version):
        report.fail(rel(PLUGIN_JSON), "", f"version {version!r} is not YYYY.MM.DD (the release tag)")
    tags = subprocess.run(["git", "tag", "--list", "v[0-9][0-9][0-9][0-9].[0-9][0-9].[0-9][0-9]"],
                          cwd=REPO_ROOT, capture_output=True, text=True).stdout.split()
    latest = max((t[1:] for t in tags), default="")
    if latest and version < latest:
        report.fail(rel(PLUGIN_JSON), "", f"version {version} is older than the latest release tag v{latest}: "
                    "run scripts/bump_plugin_version.sh " + latest)
    if not os.path.isdir(SKILLS_DIR) or not os.listdir(SKILLS_DIR):
        report.fail(rel(SKILLS_DIR), "", "no skills")
        return version
    for skill in sorted(os.listdir(SKILLS_DIR)):
        path = os.path.join(SKILLS_DIR, skill, "SKILL.md")
        if not os.path.exists(path):
            report.fail(rel(os.path.join(SKILLS_DIR, skill)), "", "directory without SKILL.md")
            continue
        text = read_text(path)
        fm, _ = parse_frontmatter(text)
        if fm is None:
            report.fail(rel(path), 1, "no YAML frontmatter")
            continue
        for key in ("name", "description", "version", "user-invocable"):
            if not fm.get(key):
                report.fail(rel(path), 1, f"frontmatter lacks {key}")
        if fm.get("name") and fm["name"] != skill:
            report.fail(rel(path), 1, f"name {fm['name']!r} != directory {skill!r}")
        if "Use when" not in fm.get("description", ""):
            report.fail(rel(path), 1, "description has no 'Use when ...' trigger sentence")
        if fm.get("version") and fm["version"] != version:
            report.fail(rel(path), 1, f"version {fm['version']} != plugin.json {version}")
        hv = HEADER_VERSION_RE.search(text)
        if not hv:
            report.fail(rel(path), "", "no '**Extension:** anofox_tabfm vYYYY.MM.DD' header line")
        elif hv.group(1) != version:
            report.fail(rel(path), text[:hv.start()].count("\n") + 1,
                        f"header says v{hv.group(1)}, plugin.json {version}")
    return version


def check_model_tables(report, models, files):
    for rel_path in MODEL_TABLE_FILES:
        if files is not None and rel_path not in files:
            continue
        path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.exists(path):
            report.fail(rel_path, "", "model table file missing")
            continue
        seen = {}
        for n, line in enumerate(read_text(path).splitlines(), 1):
            m = MODEL_ROW_RE.match(line)
            if m:
                seen.setdefault(m.group(1), n)
            h = MODELS_HEADING_RE.match(line)
            if h and int(h.group(1)) != len(models):
                report.fail(rel_path, n, f"heading says {h.group(1)} models, the registry has {len(models)}")
        # Only tables of model ids count: a row whose first cell is a model-shaped id that is
        # not a model is reported only if the file has a model table at all.
        if not seen.keys() & models:
            report.fail(rel_path, "", "no model table found")
            continue
        for m in sorted(models - seen.keys()):
            report.fail(rel_path, "", f"model table lacks built-in model {m}")


def check_model_mentions(report, rel_path, text, models):
    for n, line in enumerate(text.splitlines(), 1):
        for m in MODEL_ARG_RE.finditer(line):
            mid = m.group(1)
            if mid and mid not in models and mid not in PLACEHOLDER_MODELS:
                report.fail(rel_path, n, f"unknown model {mid!r}")


# ---------------------------------------------------------------------------- name check

def collect_name_mentions(rel_path, text):
    """(name, line) for tabfm identifiers: every one in a SKILL.md description; elsewhere
    those in code spans, fenced code, or call syntax."""
    mentions = []
    lines = text.splitlines()
    fm_end = 0
    if rel_path.endswith("SKILL.md"):
        _, fm_end = parse_frontmatter(text)
        for i in range(1, fm_end):
            mentions += [(m.group(1), i + 1) for m in IDENT_RE.finditer(lines[i])]
    in_fence = False
    for i, line in enumerate(lines):
        if i < fm_end:
            continue
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            mentions += [(m.group(1), i + 1) for m in IDENT_RE.finditer(line)]
            continue
        for span in BACKTICK_SPAN_RE.finditer(line):
            mentions += [(m.group(1), i + 1) for m in IDENT_RE.finditer(span.group(1))]
        mentions += [(m.group(1), i + 1) for m in CALL_IDENT_RE.finditer(line)]
    return mentions


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--changed-since")
    ap.add_argument("--static", action="store_true", help="no binary: manifests, versions, models only")
    ap.add_argument("--duckdb", help="DuckDB binary (default build/release/duckdb, which links the extension)")
    ap.add_argument("--extension", help="a packaged anofox_tabfm.duckdb_extension to LOAD by path")
    ap.add_argument("--duckdb-version", help="download this official CLI (with --extension)")
    ap.add_argument("--arch", default="linux_amd64")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    full_scan = not args.files and not args.changed_since
    if args.files:
        files = [os.path.relpath(os.path.abspath(f), REPO_ROOT) for f in args.files]
    elif args.changed_since:
        files = changed_md_files(args.changed_since)
    else:
        files = default_scan_files()

    report = Report()
    models = registry_models_from_source()
    check_plugin(report)
    check_model_tables(report, models, None if full_scan else set(files))
    for rel_path in files:
        check_model_mentions(report, rel_path, read_text(os.path.join(REPO_ROOT, rel_path)), models)

    if not args.static:
        tmp_cli = None
        duckdb = args.duckdb
        if not duckdb and args.duckdb_version:
            sys.path.insert(0, SCRIPT_DIR)
            from smoke_test import _download_duckdb_cli  # pyright: ignore[reportMissingImports]
            tmp_cli = tempfile.mkdtemp(prefix="doc_snippets_cli_")
            duckdb = _download_duckdb_cli(args.duckdb_version, args.arch, tmp_cli)
        duckdb = os.path.abspath(duckdb or DEFAULT_DUCKDB)
        if not os.path.exists(duckdb):
            print(f"ERROR: duckdb binary not found at {duckdb} -- build it (make release), "
                  "pass --duckdb/--extension, or run with --static", file=sys.stderr)
            return 2
        try:
            run_binary_checks(report, Runner(duckdb, args.extension and os.path.abspath(args.extension)),
                              files, models, full_scan)
        finally:
            if tmp_cli:
                shutil.rmtree(tmp_cli, ignore_errors=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in report.rows:
            f.write("\t".join(row) + "\n")
    for row in report.rows:
        if row[0] in ("FAIL", "STALE"):
            print("\t".join(row))
    order = ["PASS", "FAIL", "EXPECTED-ERROR", "SYNOPSIS", "SKIP-NETWORK", "SKIP-VERSION", "STALE"]
    print("check_doc_snippets: " + ", ".join(f"{report.counts.get(s, 0)} {s}" for s in order)
          + (" (static)" if args.static else ""))
    return 1 if report.counts.get("FAIL") or report.counts.get("STALE") else 0


def run_binary_checks(report, runner, files, models, full_scan):
    live_models = runner.query_set("SELECT model FROM tabfm_list_models();")
    if live_models != models:
        report.fail("src/tabfm_registry.cpp", "",
                    f"registry parse != tabfm_list_models(): only parsed {sorted(models - live_models)}, "
                    f"only live {sorted(live_models - models)} (fix REGISTRY_CPP_MODELS or the regex)")
    names = runner.query_set("SELECT DISTINCT function_name FROM duckdb_functions() "
                             "WHERE function_name LIKE '%tabfm%';")
    names |= runner.query_set("SELECT name FROM duckdb_settings() WHERE name LIKE '%tabfm%';")
    known = load_name_list(KNOWN_NAMES_FILE)
    removed = load_name_list(REMOVED_NAMES_FILE)
    expected_errors = load_key_set(EXPECTED_ERRORS_FILE)
    synopsis = load_key_set(SYNOPSIS_FILE)
    major = runner.duckdb_major()
    fixtures = read_text(FIXTURES_FILE) if os.path.exists(FIXTURES_FILE) else ""
    fixture_names = set(CREATE_NAME_RE.findall(strip_sql_comments(fixtures)))
    model_lit_re = re.compile(r"'(" + "|".join(re.escape(m) for m in sorted(models, key=len, reverse=True)) + r")'")

    all_keys, mentioned, seen_unknown = set(), set(), set()
    for rel_path in files:
        text = read_text(os.path.join(REPO_ROOT, rel_path))
        for name, line_no in collect_name_mentions(rel_path, text):
            mentioned.add(name)
            if name in names or name in known or name in removed or (rel_path, name) in seen_unknown:
                continue
            seen_unknown.add((rel_path, name))
            report.fail(rel_path, line_no, f"unknown function or setting name {name}")
        for name, line_no in collect_name_mentions(rel_path, text):
            if name in removed:
                report.fail(rel_path, line_no, f"{name} was removed: {removed[name]}")

        setup_accum = []
        for block, start in extract_sql_blocks(text):
            first = first_non_comment_line(block)
            key = f"{rel_path}\t{first}"
            all_keys.add(key)
            if NETWORK_RE.search(strip_sql_comments(block)):
                status, detail = "SKIP-NETWORK", "needs the network or another extension"
            elif V2_ONLY_RE.search(block) and major < 2:
                status, detail = "SKIP-VERSION", "DuckDB 2.0+ block, binary is older"
            elif key in synopsis:
                status, detail = "SYNOPSIS", ""
            else:
                sql = fixtures
                for s in select_required_setup(block, setup_accum, fixture_names):
                    sql += "\n" + s
                sql += "\n" + block
                rc, out, err = runner.run(sql, set(model_lit_re.findall(strip_sql_comments(sql))))
                if rc == 0:
                    status, detail = ("FAIL", "expected an error but it passed") if key in expected_errors \
                        else ("PASS", "")
                else:
                    status, detail = ("EXPECTED-ERROR", "") if key in expected_errors \
                        else ("FAIL", first_error_line(err, out))
            report.add(status, rel_path, start + 1, first[:100], detail)
            if SETUP_RE.search(block) and not PLACEHOLDER_ELLIPSIS_RE.search(block):
                name = setup_created_name(block)
                if name:
                    setup_accum.append((name, block))

    if full_scan:
        for key in sorted(expected_errors - all_keys):
            report.add("STALE", key.split("\t", 1)[0], "", "", f"stale expected_errors.txt entry: {key}")
        for key in sorted(synopsis - all_keys):
            report.add("STALE", key.split("\t", 1)[0], "", "", f"stale synopsis.txt entry: {key}")
        for name in sorted(removed):
            if name in names:
                report.add("STALE", "scripts/doc_snippets/removed_names.txt", "", "", f"{name} is registered again")
        for name in sorted(known):
            if name in names:
                report.add("STALE", "scripts/doc_snippets/known_names.txt", "", "",
                           f"{name} is a registered function or setting; drop it from the list")
            elif name not in mentioned:
                report.add("STALE", "scripts/doc_snippets/known_names.txt", "", "", f"{name} is mentioned nowhere")


if __name__ == "__main__":
    sys.exit(main())
