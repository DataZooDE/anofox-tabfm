"""The masked (ROCm) contract: padded + masked == the positional CPU forward, and every control is caught.

Random weights at the fixture config (offline). Upstream zero-initialises every attention output projection and
feed-forward output, which would make the attention -- and so every mask -- invisible; ``_model`` wakes them. The
same gate runs on the real weights in test_real_weights.py.
"""

from __future__ import annotations

import os
import pathlib

import pytest
import torch

from export_orion_msp import configs, export
from export_orion_msp.mask_parity import NEGATIVE_CONTROLS, TOL, check_onnx, compare
from export_orion_msp.orion_msp_patches import build_model

REPO = pathlib.Path(__file__).resolve().parents[3]
WEIGHTS = pathlib.Path(os.path.expanduser("~/.cache/anofox-tabfm/Lexsi__Orion-MSP@main/classification/model.safetensors"))


def _model():
    # two row blocks, not the fixture's one: the random links only change feature tokens, which reach the CLS
    # outputs one block later -- with a single block the random-link control could not be caught
    m = build_model({**configs.fixture().model_kwargs, "row_num_blocks": 2}, seed=3)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "rope" not in name and (p.dim() >= 2 or "out_proj" in name or "linear2" in name):
                p.copy_(torch.randn_like(p) * 0.3)
    return m


def test_padded_and_masked_equals_positional():
    assert compare(_model()) < TOL


@pytest.mark.parametrize("control", sorted(NEGATIVE_CONTROLS))
def test_every_negative_control_is_caught(control):
    m = _model()
    with NEGATIVE_CONTROLS[control]():
        assert compare(m) > 10 * TOL, control


def test_the_masked_graph_in_ort_equals_positional(tmp_path):
    import onnx

    cfg = configs.fixture()
    m = _model()
    graph = tmp_path / "mask.onnx"
    export.export_mask_graph(m, graph, dim_rows=cfg.dim_rows, dim_features=cfg.dim_features, example=cfg.example)
    res = check_onnx(graph, m)
    assert res["ok"], res
    proto = onnx.load(str(graph), load_external_data=False)
    assert [i.name for i in proto.graph.input] == ["x", "y", "train_size", "d"]
    ops = {n.op_type for n in proto.graph.node}
    assert not ops & {"SplitToSequence", "SequenceAt", "Einsum", "Unique", "NonZero", "OneHot"}, ops


@pytest.mark.skipif(not WEIGHTS.exists(), reason="Orion-MSP weights not in the cache (tabfm_download)")
def test_the_masked_rocm_contract_on_the_real_weights():
    """Padded + masked == positional on the released checkpoint, and every negative control is caught."""
    from safetensors.torch import load_file

    from export_orion_msp.mask_parity import gate

    m = build_model(configs.real().model_kwargs)
    sd = {k.removeprefix(export.CKPT_KEY_PREFIX): v for k, v in load_file(str(WEIGHTS)).items()}
    m.load_state_dict(sd, strict=True)
    res = gate(m)
    assert res["ok"], res


@pytest.mark.skipif(not WEIGHTS.exists(), reason="Orion-MSP weights not in the cache (tabfm_download)")
def test_the_bundled_migraphx_graph_in_ort_on_the_real_weights(tmp_path):
    """The committed resource (external data at fixed offsets into the cached safetensors) == positional torch."""
    import shutil

    from safetensors.torch import load_file

    graph = REPO / "resources/graph_migraphx_orion_msp_classification.onnx"
    m = build_model(configs.real().model_kwargs)
    sd = {k.removeprefix(export.CKPT_KEY_PREFIX): v for k, v in load_file(str(WEIGHTS)).items()}
    m.load_state_dict(sd, strict=True)
    # ORT resolves external data beside the graph: stage it next to a copy of the weights it references
    staged = tmp_path / graph.name
    shutil.copyfile(graph, staged)
    shutil.copyfile(WEIGHTS, tmp_path / "model.safetensors")  # ORT refuses a symlink out of the model dir
    res = check_onnx(staged, m)
    assert res["ok"], res
