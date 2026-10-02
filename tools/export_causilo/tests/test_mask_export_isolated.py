"""The masked Causilo export: does it compute the right thing, in the real flow, without the four GPU-only traps?

Sections, each answering a question the others cannot:

  1. the structural GUARDS, shown to FAIL on a deliberately bad graph (a guard never seen failing is not known to
     work) and to stay quiet on its good twin;
  2. the gate (mask_parity_full) passes, and every negative control is CAUGHT;
  3. the exported ONNX run in ONNX Runtime at padded shapes against the UNPADDED reference, every row;
  4. the real exported graphs pass all four guards, and the guards SEE the real thing (reintroduce a K=1 matmul and
     the guard goes red);
  5. independence from process-global patches, and a cross-check against upstream's own runner;
  6. a fresh-process export through the CLI, the only faithful way to reproduce the real flow.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from export_causilo import causilo_mask_forward as cmf
from export_causilo import configs, export
from export_causilo import mask_parity_full as gate
from export_causilo.causilo_mask_export import MaskExportWrapper
from export_causilo.causilo_patches import ExportWrapper, apply, build_model
from export_causilo.fixture import seeded_model, sensitive

import migraphx_guards as guards

CONFIGS = ("fixture", "fixture_r")
TASKS = ("classification", "regression")


# --- 1. the guards, against bad graphs ---------------------------------------------------------------------

def _export_tiny(module, example, path, dynamic_shapes=None):
    with torch.no_grad():
        torch.onnx.export(module.eval(), example, str(path), dynamo=True, opset_version=18,
                          dynamic_shapes=dynamic_shapes, optimize=False)
    return path


class _IntDiv(torch.nn.Module):          # `j < (n + f - 1) // f`: the TabPFN group-count bug
    def forward(self, x, n):
        j = torch.arange(8) * 1
        return x * (j < (n[0] + 2) // 3).to(x.dtype)


class _IntDivFixed(torch.nn.Module):     # the division-free spelling
    def forward(self, x, n):
        j = torch.arange(8)
        return x * (j * 3 < n[0]).to(x.dtype)


class _ScalarReshape(torch.nn.Module):
    def forward(self, x):
        return x * x.sum().reshape(())


class _ScalarReshapeFixed(torch.nn.Module):
    def forward(self, x):
        return x * x.sum().reshape(1)[0]


class _ConcatOfTransposes(torch.nn.Module):
    def forward(self, a, b):
        return torch.cat([a.transpose(0, 1), b.transpose(0, 1)], dim=0)


class _ConcatFixed(torch.nn.Module):
    def forward(self, a, b):
        return torch.cat([a, b], dim=0).transpose(0, 1)     # one Transpose, after the Concat


class _K1Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 8)

    def forward(self, x, n):
        return x * self.lin(n.reshape(1, 1))


class _K1Elementwise(_K1Matmul):
    def forward(self, x, n):
        return x * cmf._linear_k1(self.lin, n.reshape(1, 1))


class _Chunk(torch.nn.Module):             # Tensor.chunk -> SplitToSequence + SequenceAt
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 16)

    def forward(self, x):
        k, v = self.lin(x).chunk(2, dim=-1)
        return k * v


class _Sliced(_Chunk):
    def forward(self, x):
        p = self.lin(x)
        return p[..., :8] * p[..., 8:]


def _bad_and_good():
    f4 = torch.randn(4, 8)
    return {
        "integer division": (_IntDiv(), (f4, torch.tensor([8])), _IntDivFixed()),
        "reshape to []": (_ScalarReshape(), (torch.randn(1),), _ScalarReshapeFixed()),
        "concat of transposes": (_ConcatOfTransposes(), (torch.randn(3, 5), torch.randn(3, 5)), _ConcatFixed()),
        "K=1 matmul": (_K1Matmul(), (f4, torch.tensor([[2.0]])), _K1Elementwise()),
        "sequence ops": (_Chunk(), (torch.randn(3, 5, 8),), _Sliced()),
    }


@pytest.mark.parametrize("name", list(guards.GUARDS))
def test_each_guard_flags_its_bad_graph_and_not_the_fixed_one(tmp_path, name):
    bad, example, good = _bad_and_good()[name]
    guard = guards.GUARDS[name]
    assert guard(_export_tiny(bad, example, tmp_path / "bad.onnx")), f"the {name} guard did not flag a graph that has it"
    assert not guard(_export_tiny(good, example, tmp_path / "good.onnx")), f"the {name} guard flags the fixed spelling"


# --- 2. the gate ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_name", CONFIGS)
@pytest.mark.parametrize("task", TASKS)
def test_gate_passes(cfg_name, task):
    worst, rows = gate._all_cases(cfg_name, task)
    assert worst < gate.TOL, rows


@pytest.mark.parametrize("cfg_name", CONFIGS)
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("control", [c[0] for c in gate.controls()])
def test_every_negative_control_is_caught(cfg_name, task, control):
    name, install, restore = next(c for c in gate.controls() if c[0] == control)
    install()
    try:
        worst = gate._all_cases(cfg_name, task)[0]
    finally:
        restore()
    assert worst >= gate.TOL * 10, f"the gate is blind to: {name} ({cfg_name}/{task}, rel {worst:.2e})"


# --- 3 + 4. the exported graphs ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=[(c, t) for c in CONFIGS for t in TASKS], ids=lambda p: f"{p[0]}-{p[1]}")
def exported(request, tmp_path_factory):
    cfg_name, task = request.param
    cfg = configs.get(cfg_name)
    model = sensitive(build_model(task, cfg, seed=gate.SEED), weight_gain=cfg.weight_gain).eval()
    path = tmp_path_factory.mktemp(f"{cfg_name}_{task}") / "graph_mask.onnx"
    classes = cfg.classes if task == "classification" else 0
    export.export_mask_graph(model, path, example=cfg.mask_example, classes=classes)
    return cfg, task, classes, model, path


def test_ort_matches_the_unpadded_reference_on_every_row(exported):
    cfg, task, classes, model, path = exported
    r = export.check_mask_parity(path, ExportWrapper(model).eval(), cfg.mask_parity_cases, classes)
    assert r["ok"], r
    for case in r["shapes"]:
        assert case["max_abs_delta_context_rows"] < export.PARITY_TOL
        assert case["max_abs_delta_query_rows"] < export.PARITY_TOL


@pytest.mark.parametrize("guard", list(guards.GUARDS))
def test_the_exported_graph_passes_every_gpu_guard(exported, guard):
    *_, path = exported
    bad = guards.GUARDS[guard](path)
    assert not bad, f"{guard}: {bad[:5]}"


def test_the_graph_declares_the_five_masked_inputs(exported):
    import onnx
    *_, path = exported
    m = onnx.load(str(path), load_external_data=False)
    assert [i.name for i in m.graph.input] == ["x", "y", "train_size", "n_rows", "d"]


def test_the_k1_guard_sees_the_real_graph(tmp_path, monkeypatch):
    """Reintroduce `Linear(1, n)` as a matmul and the guard must go red on the REAL masked graph. This is the
    evidence the guard is looking at something that exists there: QueryScale has two such layers per attention."""
    monkeypatch.setattr(cmf, "_linear_k1", lambda lin, x: F.linear(x, lin.weight, lin.bias))
    cfg = configs.get("fixture")
    model = seeded_model("classification")
    path = tmp_path / "k1.onnx"
    export.export_mask_graph(model, path, example=cfg.mask_example, classes=cfg.classes)
    assert guards.k1_matmuls(path), "no K=1 matmul found even though one was reintroduced"


def test_the_sequence_guard_sees_the_real_graph(tmp_path, monkeypatch):
    """Put upstream's `chunk` back and the guard must go red on the REAL masked graph (it is where this was found)."""
    def chunked(att, rows):
        packed = F.linear(rows, att.projection.weight[att.width:], att.projection.bias[att.width:])
        return att._keys(*packed.chunk(2, dim=-1))

    monkeypatch.setattr(cmf, "_prepare_context", chunked)
    cfg = configs.get("fixture")
    path = tmp_path / "chunk.onnx"
    export.export_mask_graph(seeded_model("classification"), path, example=cfg.mask_example, classes=cfg.classes)
    assert guards.sequence_ops(path), "no sequence op found even though chunk was reintroduced"


# --- 5. independence from patches, and upstream's own runner ----------------------------------------------

def test_the_masked_path_does_not_call_the_patched_upstream_methods(monkeypatch):
    """Everything `causilo_patches.apply()` replaces must be unreachable from the masked forward. If it is, the
    masked graph cannot depend on which process patched what (the trap the TabPFN 2.5 path fell into)."""
    from causilo.nn import embeddings, prediction
    from causilo.nn.layers import scaling

    model = seeded_model("classification")
    cfg = configs.get("fixture")
    n, N, d, T, H = cfg.mask_parity_cases[0]
    rng = np.random.default_rng(0)
    feed = export.padded_feed(rng.standard_normal((n, d)), rng.integers(0, 3, N), (T, H))
    args = [torch.from_numpy(feed[k]) for k in ("x", "y", "train_size", "n_rows", "d")]
    with torch.no_grad():
        before = MaskExportWrapper(model)(*args)

    def forbidden(*_a, **_k):
        raise AssertionError("the masked forward reached a method that apply() patches")

    for cls, attr in ((scaling.QueryScale, "forward"), (prediction.Head, "forward"),
                      (embeddings.TargetEmbedding, "forward")):
        monkeypatch.setattr(cls, attr, forbidden)
    with torch.no_grad():
        after = MaskExportWrapper(model)(*args)
    assert torch.equal(before, after)


def test_masked_matches_upstreams_own_runner_and_numpy_normaliser():
    """The reference of the gate is ExportWrapper; this ties the masked path straight to upstream's code."""
    from causilo.data.normalization import Normalizer
    from causilo.execution.runner import ModelRunner

    apply()
    cfg = configs.get("fixture")
    model = seeded_model("classification")
    n, N, d, T, H = 24, 10, 5, 48, 16
    g = np.random.default_rng(3)
    x = (g.standard_normal((n, d)) * g.uniform(0.5, 4, d) + g.uniform(-2, 2, d)).astype(np.float32)
    y = g.integers(0, 3, N)
    table = Normalizer.fit(x[:N].copy(), "none").transform(x)                  # upstream's numpy normaliser
    both = torch.from_numpy(np.concatenate([table[:N], table], 0))[None]       # context, then every row
    with torch.no_grad():
        want = ModelRunner(model).predict(both, torch.from_numpy(y.astype(np.float32))[None])[0].numpy()
        feed = export.padded_feed(x, y, (T, H))
        got = MaskExportWrapper(model)(*[torch.from_numpy(feed[k]) for k in
                                         ("x", "y", "train_size", "n_rows", "d")])[0, :n].numpy()
    assert cfg.classes == got.shape[1]
    assert np.abs(got - want).max() < 1e-4


# --- MIGraphX's own parser (local only: needs migraphx-driver and, for the weights, nothing but this machine) ----

@pytest.mark.skipif(shutil.which("migraphx-driver") is None, reason="needs migraphx-driver (ROCm)")
@pytest.mark.parametrize("task", TASKS)
def test_migraphx_parses_and_reference_compiles_the_masked_graph(tmp_path, task):
    """What this CAN see: parse-time failures (`Unknown operator`, a Reshape to []) and anything the REFERENCE
    target's passes reject. What it CANNOT see, measured: the GPU pipeline's `simplify_reshapes` failures, which the
    reference target passes (the known-bad TabPFN-3 repro compiles on --ref) and for which there is no
    GPU-free target here (libmigraphx_cpu.so is not installed). So a pass is necessary, not sufficient."""
    import json
    import subprocess

    from safetensors.torch import save_file

    cfg = configs.get("fixture")
    model = seeded_model(task)
    classes = cfg.classes if task == "classification" else 0
    graph = tmp_path / f"graph_mask_causilo_{task}.onnx"
    export.export_mask_graph(model, graph, example=cfg.mask_example, classes=classes)
    tensor_map = export.postprocess(graph, dict(model.state_dict()))
    tmap = tmp_path / f"tensor_map_causilo_{task}.json"
    export.write_tensor_map(tmap, tensor_map, task=task, safetensors_rel="model.safetensors")
    save_file({k: v.detach().contiguous() for k, v in sorted(model.state_dict().items())},
              str(tmp_path / "model.safetensors"))
    out = tmp_path / "graph_migraphx.onnx"
    tool = pathlib.Path(__file__).resolve().parents[2] / "make_migraphx_graph.py"
    r = subprocess.run([sys.executable, str(tool), task, "--weights", str(tmp_path / "model.safetensors"),
                        "--graph", str(graph), "--tensor-map", str(tmap), "--dynamic", "--out", str(out)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:] + r.stdout[-500:]
    r = subprocess.run(["migraphx-driver", "compile", str(out), "--onnx", "--ref", "--input-dim", "@x", "1", "64", "16",
                        "@y", "1", "64", "@train_size", "1", "@n_rows", "1", "@d", "1"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stderr + r.stdout)[-1500:]


# --- 6. a fresh process -----------------------------------------------------------------------------------

@pytest.mark.parametrize("task", TASKS)
def test_masked_export_in_a_fresh_process(tmp_path, task):
    r = subprocess.run(
        [sys.executable, "-m", "export_causilo.cli", "--task", task, "--config", "fixture", "--contract", "mask",
         "--out", str(tmp_path)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-1500:] + r.stdout[-500:]
    graph = tmp_path / f"graph_mask_causilo_{task}.onnx"
    assert graph.exists()
    for name, guard in guards.GUARDS.items():
        assert not guard(graph), f"{name}: {guard(graph)[:5]}"
