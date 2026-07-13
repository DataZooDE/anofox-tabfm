"""DuckDB session helper for the tabfm TUI.

All extension interaction lives here (the UI layer stays thin). The extension
itself is a native .duckdb_extension — this module only *loads* it; there is no
Python dependency inside the extension.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[4]


def find_extension() -> Path:
    """Locate the built anofox_tabfm loadable extension.

    Honors $TABFM_EXT_PATH; otherwise searches the usual build trees.
    """
    env = os.environ.get("TABFM_EXT_PATH")
    if env:
        return Path(env)
    candidates = [
        REPO_ROOT / "build" / cfg / "extension" / "anofox_tabfm" / "anofox_tabfm.duckdb_extension"
        for cfg in ("release", "debug", "reldebug")
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find anofox_tabfm.duckdb_extension. Build it (make release) "
        "or set TABFM_EXT_PATH to the .duckdb_extension path."
    )


@dataclass
class ModelRow:
    model: str
    family: str
    capabilities: str
    license: str
    commercial: bool
    downloaded: bool


class TabfmSession:
    """A live DuckDB connection with anofox_tabfm loaded."""

    def __init__(self, ext_path: Path | None = None):
        self.ext_path = ext_path or find_extension()
        self.con = duckdb.connect(
            ":memory:", config={"allow_unsigned_extensions": "true"}
        )
        # httpfs for hf:// datasets; ignore if already present / offline.
        for stmt in ("INSTALL httpfs", "LOAD httpfs"):
            try:
                self.con.execute(stmt)
            except duckdb.Error:
                pass
        self.con.execute(f"LOAD '{self.ext_path.as_posix()}'")

    # --- discovery -------------------------------------------------------
    def list_models(self) -> list[ModelRow]:
        """The model registry. Falls back gracefully if tabfm_list_models()
        is not in this build yet (pre-registry surface)."""
        try:
            rows = self.con.execute(
                "SELECT model, family, capabilities::VARCHAR, license, "
                "commercial, downloaded FROM tabfm_list_models() ORDER BY model"
            ).fetchall()
            return [ModelRow(*r) for r in rows]
        except duckdb.Error:
            pass
        # pre-registry: derive a single default entry from tabfm_models()
        try:
            rows = self.con.execute(
                "SELECT DISTINCT model, license, loaded FROM tabfm_models()"
            ).fetchall()
            return [
                ModelRow(r[0], "icl-transformer", "classify,regress", r[1], False, bool(r[2]))
                for r in rows
            ]
        except duckdb.Error:
            return [ModelRow("tabfm-v1", "icl-transformer", "classify,regress", "?", False, False)]

    def settings(self) -> list[tuple[str, str]]:
        rows = self.con.execute(
            "SELECT name, value FROM duckdb_settings() WHERE name LIKE 'anofox_tabfm_%' ORDER BY name"
        ).fetchall()
        return [(str(a), str(b)) for a, b in rows]

    # --- prediction ------------------------------------------------------
    def predict(
        self,
        data: str,
        target: str,
        task: str = "classify",
        model: str | None = None,
        test: str | None = None,
        limit: int = 200,
    ):
        fn = "tabfm_classify" if task == "classify" else "tabfm_regress"
        args = [self._lit(data), self._lit(target)]
        if test:
            args.append(f"test := {self._lit(test)}")
        if model:
            args.append(f"model := {self._lit(model)}")
        sql = f"SELECT * FROM {fn}({', '.join(args)}) LIMIT {int(limit)}"
        cur = self.con.execute(sql)
        cols = [d[0] for d in cur.description]
        return cols, cur.fetchall(), sql

    @staticmethod
    def _lit(s: str) -> str:
        return "'" + s.replace("'", "''") + "'"

    def close(self):
        self.con.close()
