"""TabICL v2 on the REAL checkpoints: the shipped context-cache pair against the combined graph.

Skipped unless TABFM_REAL_WEIGHTS is set. Reads the model.safetensors the engine uses
(~/.cache/anofox-tabfm/jingang__TabICL@4dcd344ece2c00be9e831fdd35bed57b5ad83e19/<task>/, written by convert_weights.py or tabfm_download), never
anything in this repo.
"""

from __future__ import annotations

import os
import pathlib

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TABFM_REAL_WEIGHTS"),
                                reason="needs the real TabICL v2 checkpoints (set TABFM_REAL_WEIGHTS=1)")

CACHE = pathlib.Path(os.environ.get("TABFM_CACHE", "~/.cache/anofox-tabfm")).expanduser() / "jingang__TabICL@4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
RESOURCES = pathlib.Path(__file__).resolve().parents[3] / "resources"
TASKS = ("classification", "regression")


@pytest.mark.parametrize("task", TASKS)
def test_the_shipped_context_cache_pair_on_the_real_weights(task):
    """The bundled kv pair (resources/graph_{prepare,query}_tabicl_<task>.onnx) against the bundled combined
    graph, real weights injected through each graph's tensor map: identical labels and |dlogit| <= 1e-4
    (classification), |dyhat| <= 1e-4 x std(y) (regression), on every row, context rows included; re-batching
    leaves the answers alone; and a permuted-label control moves them, so the identity is not vacuous."""
    from export_tabicl import kv

    prep, qry = kv.pair_paths(RESOURCES, task)
    res = kv.gate(prep, qry, RESOURCES / f"graph_tabicl_{task}.onnx", CACHE / task / "model.safetensors", task)
    assert res["ok"], res["failures"]
