"""Tests for the tabfm TUI — real extension + committed fixture (no mocks).

Skips cleanly if the extension isn't built yet.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.widgets import DataTable, Input

from tabfm_tui.app import TabfmTui
from tabfm_tui.session import TabfmSession, find_extension

REPO = Path(__file__).resolve().parents[3]
FIXTURE_DIR = (REPO / "test" / "fixtures" / "multi").as_posix()

def _ext_built() -> bool:
    try:
        find_extension()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _ext_built(), reason="anofox_tabfm extension not built")


def test_session_lists_registry():
    s = TabfmSession()
    try:
        models = {m.model for m in s.list_models()}
        assert "tabfm-v1" in models
    finally:
        s.close()


def test_app_mounts_lists_and_predicts():
    async def run():
        app = TabfmTui()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # the registry populated the models table on mount
            assert app.query_one("#models", DataTable).row_count >= 1
            # a quick-pick demo fills the form
            await pilot.click("#demo_iris")
            assert "iris" in app.query_one("#data", Input).value
            # a real fixture forward pass via model := (offline, deterministic)
            app.session.con.execute(f"SET anofox_tabfm_model_manifest='{FIXTURE_DIR}'")
            app.session.con.execute(
                "CREATE TABLE fx AS SELECT * FROM (VALUES "
                "(0.5,1.0,'a','x',0.1,'c0'),(1.5,0.2,'b','y',0.3,NULL)) v(f1,f2,f3,f4,f5,label)"
            )
            cols, rows, _ = app.session.predict("fx", "label", "classify", model="fixture-commercial")
            assert "yhat" in cols and len(rows) == 2

    asyncio.run(run())
