"""TabICL v2 on the REAL checkpoints: the masked (ROCm) contract against the positional CPU graph.

Skipped unless TABFM_REAL_WEIGHTS is set. Reads the model.safetensors the engine uses
(~/.cache/anofox-tabfm/jingang__TabICL@main/<task>/, written by convert_weights.py or tabfm_download), never
anything in this repo. The masked graph is the one export_tabicl --contract mask writes; set
TABICL_MASK_DIR to its --out directory to also compare it, in ORT, against the SHIPPED positional graph.
"""

from __future__ import annotations

import os
import pathlib

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TABFM_REAL_WEIGHTS"),
                                reason="needs the real TabICL v2 checkpoints (set TABFM_REAL_WEIGHTS=1)")

CACHE = pathlib.Path(os.environ.get("TABFM_CACHE", "~/.cache/anofox-tabfm")).expanduser() / "jingang__TabICL@main"
RESOURCES = pathlib.Path(__file__).resolve().parents[3] / "resources"
TASKS = ("classification", "regression")


def _real_model(task):
    from safetensors.torch import load_file

    from export_tabicl import configs
    from export_tabicl.tabicl_patches import build_model

    model = build_model(task, configs.real(task).model_kwargs, seed=0)
    missing, unexpected = model.load_state_dict(load_file(str(CACHE / task / "model.safetensors")), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return model


@pytest.mark.parametrize("task", TASKS)
def test_the_masked_contract_on_the_real_weights(task):
    """Padded + masked == positional on the real weights, and every negative control is caught."""
    from export_tabicl.mask_parity import gate

    res = gate(_real_model(task))
    assert res["ok"], res


@pytest.mark.parametrize("task", TASKS)
def test_the_masked_graph_equals_the_shipped_positional_graph(task):
    from export_tabicl.mask_parity import check_shipped

    mask_dir = os.environ.get("TABICL_MASK_DIR")
    if not mask_dir:
        pytest.skip("set TABICL_MASK_DIR to an export_tabicl --contract mask --out directory")
    mask_dir = pathlib.Path(mask_dir)
    res = check_shipped(mask_dir / f"graph_mask_tabicl_{task}.onnx", mask_dir / f"tensor_map_mask_tabicl_{task}.json",
                        RESOURCES / f"graph_tabicl_{task}.onnx", RESOURCES / f"tensor_map_tabicl_{task}.json",
                        CACHE / task / "model.safetensors", _real_model(task))
    assert res["ok"], res
