"""A Textual TUI to try the anofox-tabfm DuckDB extension quickly.

    uv run tabfm-tui           # or: TABFM_EXT_PATH=... uv run tabfm-tui

Left: the model registry + settings. Right: a predict form (dataset, target,
task, model) with quick-pick Hugging Face demos and a results table.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    RadioButton,
    RadioSet,
    Select,
    Static,
)

from .session import TabfmSession

# Quick-pick demos — the same hf:// datasets as examples/ (offline-friendly note:
# these need network + the downloaded model weights).
DEMOS = {
    "iris (3-class)": ("hf://datasets/scikit-learn/iris/**/*.csv", "Species", "classify"),
    "churn (binary)": ("hf://datasets/scikit-learn/churn-prediction/**/*.csv", "Churn", "classify"),
    "tips (regression)": ("hf://datasets/scikit-learn/tips/**/*.csv", "tip", "regress"),
}


class TabfmTui(App):
    CSS = """
    #sidebar { width: 42; border-right: solid $primary; }
    #main { padding: 0 1; }
    .h { text-style: bold; color: $secondary; padding: 1 0 0 0; }
    #status { color: $text-muted; }
    #err { color: $error; }
    DataTable { height: 1fr; }
    Input, Select, RadioSet { margin: 0 0 1 0; }
    """
    BINDINGS = [("q", "quit", "Quit"), ("r", "run", "Run predict")]

    def __init__(self):
        super().__init__()
        self.session: TabfmSession | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with VerticalScroll(id="sidebar"):
                yield Static("Models (registry)", classes="h")
                yield DataTable(id="models", zebra_stripes=True)
                yield Static("Settings", classes="h")
                yield DataTable(id="settings", zebra_stripes=True)
            with Vertical(id="main"):
                yield Static("Predict", classes="h")
                yield Input(placeholder="dataset (table name or hf://… path)", id="data")
                yield Input(placeholder="target column", id="target")
                with RadioSet(id="task"):
                    yield RadioButton("classify", value=True)
                    yield RadioButton("regress")
                yield Select([], prompt="model := (default)", id="model", allow_blank=True)
                yield Input(placeholder="test relation (optional, train/test split)", id="test")
                with Horizontal():
                    for name in DEMOS:
                        yield Button(name, id="demo_" + name.split()[0])
                    yield Button("Run predict", variant="primary", id="run")
                yield Static("", id="status")
                yield Static("", id="err")
                yield DataTable(id="results", zebra_stripes=True)
        yield Footer()

    # --- lifecycle -------------------------------------------------------
    def on_mount(self) -> None:
        try:
            self.session = TabfmSession()
        except Exception as exc:  # noqa: BLE001 — surface any load failure
            self.query_one("#err", Static).update(f"⚠ {exc}")
            return
        self.query_one("#status", Static).update(f"loaded {self.session.ext_path.name}")
        self._refresh_models()
        self._refresh_settings()

    def _refresh_models(self) -> None:
        table = self.query_one("#models", DataTable)
        table.clear(columns=True)
        table.add_columns("model", "caps", "comm", "dl")
        options = []
        for m in self.session.list_models():
            table.add_row(m.model, m.capabilities, "✓" if m.commercial else "·",
                          "✓" if m.downloaded else "·")
            options.append((m.model, m.model))
        self.query_one("#model", Select).set_options(options)

    def _refresh_settings(self) -> None:
        table = self.query_one("#settings", DataTable)
        table.clear(columns=True)
        table.add_columns("setting", "value")
        for name, value in self.session.settings():
            table.add_row(name.replace("anofox_tabfm_", ""), value or "·")

    # --- actions ---------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run":
            self.action_run()
            return
        for name, (data, target, task) in DEMOS.items():
            if event.button.id == "demo_" + name.split()[0]:
                self.query_one("#data", Input).value = data
                self.query_one("#target", Input).value = target
                buttons = list(self.query_one("#task", RadioSet).query(RadioButton))
                buttons[0 if task == "classify" else 1].value = True
                return

    def action_run(self) -> None:
        if not self.session:
            return
        data = self.query_one("#data", Input).value.strip()
        target = self.query_one("#target", Input).value.strip()
        if not data or not target:
            self.query_one("#err", Static).update("⚠ need a dataset and a target column")
            return
        task = "classify" if self.query_one("#task", RadioSet).pressed_index == 0 else "regress"
        model = self.query_one("#model", Select).value
        model = None if (model is Select.BLANK or not model) else str(model)
        test = self.query_one("#test", Input).value.strip() or None
        self.query_one("#err", Static).update("")
        self.query_one("#status", Static).update("running… (a real forward pass can take ~1 min)")
        self.run_worker(
            lambda: self._do_predict(data, target, task, model, test), thread=True, exclusive=True
        )

    def _do_predict(self, data, target, task, model, test) -> None:
        try:
            cols, rows, sql = self.session.predict(data, target, task, model, test)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self.query_one("#err", Static).update, f"⚠ {exc}")
            self.call_from_thread(self.query_one("#status", Static).update, "")
            return
        self.call_from_thread(self._show_results, cols, rows, sql)

    def _show_results(self, cols, rows, sql) -> None:
        table = self.query_one("#results", DataTable)
        table.clear(columns=True)
        table.add_columns(*cols)
        for r in rows:
            table.add_row(*[("·" if v is None else str(v)) for v in r])
        self.query_one("#status", Static).update(f"{len(rows)} rows · {sql}")


def main() -> None:
    TabfmTui().run()


if __name__ == "__main__":
    main()
