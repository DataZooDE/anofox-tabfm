"""Export each architecture's masked graph in a FRESH process.

`mask_parity_full` patches all three architecture modules up front, so it cannot
see a forward that only works when the module it reaches into was the one
patched. That is exactly what happened: the 2.5 branch called
`v26.select_features`, `apply_module_patches("v2.5")` patches v2.5's module, and a
v2.5-only export hit upstream's data-dependent `torch.all(sel)`. The gate passed;
the real flow failed. A subprocess per architecture is the only faithful way to
reproduce the real flow.
"""
import subprocess
import sys

import pytest

CASES = [("fixture", "v2"), ("fixture25", "v2.5"), ("fixture26", "v2.6"),
         ("fixture3", "v3"), ("fixture3r", "v3")]


@pytest.mark.parametrize("config,arch", CASES)
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_masked_export_in_a_fresh_process(tmp_path, config, arch, task):
    r = subprocess.run(
        [sys.executable, "-m", "export_tabpfn.cli", "--task", task, "--config", config,
         "--contract", "mask", "--skip-parity", "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"{arch} {task} masked export failed in isolation:\n{r.stderr[-1500:]}"


# ---------------------------------------------------------------------------
# No integer division in anything the GPU evaluates.
#
# MIGraphX's GPU target gets `j < (n + f - 1) // f` WRONG -- for n=8, f=3 it marks
# four groups real instead of three -- while ORT and MIGraphX's own reference
# target are right. The parity gate never runs on a GPU, so nothing else in this
# suite can see it; it surfaced as a 1-3% relative error in the final logits and
# was found only by bisecting every intermediate tensor. This is the structural
# guard: an integer Div in a masked graph is the pattern, whatever it feeds.

def int_divs(path):
    """The shared guard, loaded from tools/ -- one definition, so CI's check of the
    committed graphs and this check of freshly exported ones cannot drift apart."""
    import importlib.util
    import pathlib

    tool = pathlib.Path(__file__).resolve().parents[2] / "check_migraphx_int_div.py"
    spec = importlib.util.spec_from_file_location("check_migraphx_int_div", tool)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.int_divs(path)


@pytest.mark.parametrize("config,arch", CASES)
def test_masked_graph_has_no_integer_division(tmp_path, config, arch):
    r = subprocess.run(
        [sys.executable, "-m", "export_tabpfn.cli", "--task", "classification", "--config", config,
         "--contract", "mask", "--skip-parity", "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-1500:]
    g = next(tmp_path.glob("graph_mask_*.onnx"))
    bad = int_divs(g)
    assert not bad, (f"{arch}: integer Div in a masked graph {bad[:4]} -- MIGraphX's GPU target "
                     f"miscomputes floor-division-derived comparisons; use a multiply/float form")


# ---------------------------------------------------------------------------
# No Reshape to an empty shape.
#
# `t.reshape(())` exports as a Reshape whose target is [], and MIGraphX reads that
# as ZERO elements and refuses to parse the graph ("reshape has 0 elements whereas
# the input has 1"). ORT accepts it, so neither the parity gate nor an ORT run can
# see it -- it appears only when the GPU backend first loads the graph. It was hit
# once for the 2.5 line and again for v3. Index a scalar with `[0]` instead.

def scalar_reshapes(path):
    import onnx
    import math

    m = onnx.load(str(path), load_external_data=False)
    # A shape can be an initializer or a Constant node, and a Constant carries its
    # value as a tensor (`value`) OR as a list (`value_ints`). The empty target of
    # `reshape(())` arrives as `value_ints = []`; reading only `value` made this
    # guard pass on the very graph MIGraphX rejected.
    # Dims, not values: the weights are external data and may be deleted.
    sizes = {i.name: math.prod(i.dims) for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    sizes[n.output[0]] = math.prod(a.t.dims)
                elif a.name in ("value_ints", "value_floats", "value_strings"):
                    sizes[n.output[0]] = len(getattr(a, a.name[len("value_"):]))
                elif a.name in ("value_int", "value_float", "value_string"):
                    sizes[n.output[0]] = 1
    return [n.name or n.output[0] for n in m.graph.node
            if n.op_type == "Reshape" and sizes.get(n.input[1]) == 0]


@pytest.mark.parametrize("config,arch", CASES)
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_masked_graph_has_no_scalar_reshape(tmp_path, config, arch, task):
    r = subprocess.run(
        [sys.executable, "-m", "export_tabpfn.cli", "--task", task, "--config", config,
         "--contract", "mask", "--skip-parity", "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-1500:]
    g = next(tmp_path.glob("graph_mask_*.onnx"))
    bad = scalar_reshapes(g)
    assert not bad, f"{arch} {task}: Reshape to [] (MIGraphX reads it as 0 elements): {bad[:5]}"
