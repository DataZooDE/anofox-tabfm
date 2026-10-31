"""MIGraphX structural guards on freshly exported 3.5 masked graphs.

Each guard is a GPU-only failure found on an earlier TabPFN generation that ORT
and the PyTorch parity gate cannot see (docs/ROCM_TABPFN_PLAN.md):

  * an integer Div on runtime data (MIGraphX's GPU target miscomputes it);
  * a Reshape to [] (MIGraphX reads it as zero elements and refuses the graph);
  * a MatMul with a contraction dim of 1 (kills `simplify_reshapes`).

3.5 adds a binary search and integer index arithmetic to the graph (the ECDF),
which is exactly where an integer Div would creep in. The checks are the ones
tools/export_tabpfn applies, loaded from there so they cannot drift. The export
runs in a FRESH process, so the graph cannot depend on anything this process
patched.
"""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INT_DIV = _load(ROOT / "tools" / "check_migraphx_int_div.py", "check_migraphx_int_div")
ISO = _load(ROOT / "tools" / "export_tabpfn" / "tests" / "test_mask_export_isolated.py", "mask_iso")


@pytest.fixture(scope="module", params=["classification", "regression"])
def masked_graph(request, tmp_path_factory):
    out = tmp_path_factory.mktemp(request.param)
    r = subprocess.run(
        [sys.executable, "-m", "export_tabpfn35.cli", "--task", request.param,
         "--config", "fixture35", "--contract", "mask", "--skip-parity", "--out", str(out)],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-1500:]
    return next(out.glob("graph_mask_*.onnx"))


def test_no_integer_division(masked_graph):
    assert not INT_DIV.int_divs(masked_graph)


def test_no_scalar_reshape(masked_graph):
    assert not ISO.scalar_reshapes(masked_graph)


def test_no_k1_matmul(masked_graph):
    assert not ISO.k1_matmuls(masked_graph)


def test_declares_the_masked_contract(masked_graph):
    import onnx

    m = onnx.load(str(masked_graph), load_external_data=False)
    assert [i.name for i in m.graph.input] == ["x", "y", "train_size", "n_rows", "d"]
