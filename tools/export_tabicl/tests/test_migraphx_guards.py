"""The shipped ROCm graphs pass every structural MIGraphX guard (tests/migraphx_guards.py, shared with
tools/export_causilo, where each guard is exercised against a deliberately bad graph). These failures are
invisible to ONNX Runtime; they surface only when the GPU target compiles, after minutes."""

from __future__ import annotations

import pathlib

import pytest

import migraphx_guards as g

RESOURCES = pathlib.Path(__file__).resolve().parents[3] / "resources"
GUARDS = ("int_divs", "scalar_reshapes", "concat_of_transposes", "k1_matmuls", "repeated_concat_inputs",
          "sequence_ops")


@pytest.mark.parametrize("task", ("classification", "regression"))
@pytest.mark.parametrize("guard", GUARDS)
def test_the_shipped_rocm_graph_passes(task, guard):
    graph = RESOURCES / f"graph_migraphx_tabicl_{task}.onnx"
    assert getattr(g, guard)(graph) == []
