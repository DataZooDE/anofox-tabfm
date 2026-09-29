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

CASES = [("fixture", "v2"), ("fixture25", "v2.5"), ("fixture26", "v2.6")]


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

_INT = None


def int_divs(path):
    """Integer Divs that depend on RUNTIME DATA (x, y, train_size, n_rows, d).

    Shape arithmetic is exempt: `num_padded_columns // features_per_group` is an
    integer Div too, but its inputs come from Shape ops, which MIGraphX folds to
    constants once the bucket shape is pinned -- it never reaches the GPU as a
    computation. Only a Div reachable from a graph input through non-Shape ops
    is evaluated on the device, so only that is the bug's pattern.
    """
    import onnx
    from onnx import TensorProto as TP
    from onnx import shape_inference

    ints = {TP.INT64, TP.INT32, TP.INT16, TP.INT8, TP.UINT8, TP.UINT16, TP.UINT32, TP.UINT64}
    m = onnx.load(str(path), load_external_data=False)
    inferred = shape_inference.infer_shapes(m)
    ty = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input):
        if vi.type.HasField("tensor_type"):
            ty[vi.name] = vi.type.tensor_type.elem_type
    for init in m.graph.initializer:
        ty[init.name] = init.data_type
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    ty[n.output[0]] = a.t.data_type

    tainted = {i.name for i in m.graph.input}
    for n in m.graph.node:                      # nodes are in topological order
        if n.op_type in ("Shape", "Size"):
            continue                            # shape-only: constant at compile time
        if any(i in tainted for i in n.input):
            tainted.update(n.output)
    return [(n.name, n.output[0]) for n in m.graph.node
            if n.op_type == "Div"
            and any(i in tainted for i in n.input)
            and any(ty.get(i) in ints for i in n.input)]


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
