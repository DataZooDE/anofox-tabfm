"""CI fixture model for LimiX-2M: random weights, tiny dims, deterministic bytes.

Same shape as the Causilo / TabICL fixtures: a tiny random-init LimiX (Apache-2.0 architecture, OUR
weights, zero StableAI checkpoint bytes: see CLAUDE.md "license wall"), weight-free graphs through the
same exporter, and one v2 manifest carrying both tasks. Unlike Causilo's, ONE weights file serves both
tasks, because that is what LimiX is: a single model whose task is chosen by the export wrapper.

What had to be learned the hard way (each number was measured, see tests/test_fixture.py):

  * LimiX's numeric tokenizer encodes a value as (sign, decimal exponent, mantissa) and an EXACT zero as
    its own token. A feature value that standardises to exactly 0 flips to a different embedding under
    any rounding noise; a value near zero is hypersensitive in proportion to 1/|z|. With the obvious
    golden inputs (quarter-multiples) 41 standardised values were exactly 0, and a 1e-6 relative input
    change moved the logits by 0.04 .. 20 on a spread of 0.03 .. 12: no golden value could be stable
    across platforms, and ORT and PyTorch would "legitimately" disagree. Adding a sixteenths pattern
    (still exact in float32) breaks the exact-mean ties and brings the same weights to 1e-6 .. 6e-4.
    ``assert_well_conditioned`` keeps it that way, and rejects the quarter-multiple inputs (tested).
  * The default initialisation is nearly input-blind (logit spread 0.03, label sensitivity 0.03, one
    predicted class). Turning up ONLY the target-side projections (TARGET_GAIN) makes the output depend
    on the training labels without sharpening attention; with SEED 18 it predicts all three classes with
    a minimum top-1/top-2 margin of 0.33 against ~3e-4 of noise.
  * The golden inputs are exact in binary floating point and expressible in SQL, so the SQL test builds
    the same table and pins a prediction for EVERY row, context and query. ``golden_<task>.json`` carries
    the expected values; ``sql_inputs`` is the formula the test repeats.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import torch
from safetensors.torch import save_file

from export_limix import configs as x_configs
from export_limix import export as x_export
from export_limix.limix_patches import ExportWrapper, build_model

SEED_WEIGHTS = 18
TARGET_GAIN = 10.0                         # label embedding / y encoders / y decoders; see the docstring
MIN_LABEL_SENSITIVITY = 0.1                # output change when one training label changes
MAX_NOISE_RESPONSE = 1e-2                  # output change per 1e-6 relative input perturbation
MIN_REAL_VALUE = 0.02                      # smallest |standardised feature value| (tokenizer 1/|z| exposure)
GOLDEN_SHAPE = dict(t=20, h=5, s=12)       # T rows, H features, S train rows
PARITY_RTOL = 1e-4
MIN_MARGIN = 0.05                          # top-1 minus top-2 logit, every row: no near-ties

TASKS = ("classification", "regression")
FILES = [
    "graph_limix_classification.onnx", "tensor_map_limix_classification.json", "golden_classification.json",
    "graph_limix_regression.onnx", "tensor_map_limix_regression.json", "golden_regression.json",
    "model.safetensors", "manifest.json",
]


def sql_inputs(task: str):
    """(x [T,H], y [S]) as nested lists, from formulas the SQL test repeats verbatim.

    x[i][j] = ((i*7 + j*3 + (i*j) % 5) % 11) * 0.25 - 1.0 + ((i*13 + j*5 + (i*j) % 3) % 7) * 0.0625
    classification y[i] = (i*5 + i//3) % 3                     labels 'c0','c1','c2'
    regression     y[i] = ((i*37) % 17) * 0.25 - 2.0           exact in float32
    """
    t, h, s = (GOLDEN_SHAPE[k] for k in ("t", "h", "s"))
    x = [[((i * 7 + j * 3 + (i * j) % 5) % 11) * 0.25 - 1.0 + ((i * 13 + j * 5 + (i * j) % 3) % 7) * 0.0625
          for j in range(h)] for i in range(t)]
    if task == "classification":
        y = [float((i * 5 + i // 3) % 3) for i in range(s)]
    else:
        y = [((i * 37) % 17) * 0.25 - 2.0 for i in range(s)]
    return x, y


def golden_tensors(task: str):
    xs, ys = sql_inputs(task)
    return torch.tensor([xs], dtype=torch.float32), torch.tensor([ys], dtype=torch.float32)


def sensitive(model):
    """Scale ONLY the target-side projections of a default-initialised model (see the docstring). Used by
    the fixture AND by tests that need a model that responds to its input."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim < 2 or any(t in name for t in ("norm", "mask_embedding", "sign_embedding",
                                                          "digit_embedding")):
                continue
            if any(t in name for t in ("y_decoder", "y_encoder", "y_embedding", "y_mask")):
                param.mul_(TARGET_GAIN)
    return model


def seeded_model():
    # build_model seeds torch before constructing, so two calls give identical weights.
    return sensitive(build_model(x_configs.fixture().model_config, seed=SEED_WEIGHTS)).eval()


def _real_standardised(model, wrapper, x, y):
    """The tokenizer's own input for the REAL feature columns (padding slots excluded), plus the output."""
    store = {}
    tok = dict(model.named_modules())["encoder_x.0.numeric_mlp"]
    hook = tok.register_forward_pre_hook(lambda mod, inp: store.__setitem__("x", inp[0].detach().double()))
    try:
        with torch.no_grad():
            out = wrapper(x, y)
    finally:
        hook.remove()
    z = store["x"]                                           # [1, rows, groups, per_group, 1]
    z = z.reshape(z.shape[0], z.shape[1], -1)[..., : x.shape[2]]
    return z, out


def assert_well_conditioned(wrapper, model, x, y, task: str) -> None:
    """The fixture must respond to its input, must NOT amplify rounding noise, and must not contain a
    feature value the tokenizer is singular at (an exact zero, or one closer to zero than MIN_REAL_VALUE)."""
    z, base = _real_standardised(model, wrapper, x, y)
    zeros = int((z == 0).sum())
    assert zeros == 0, (
        f"{zeros} standardised feature value(s) are an exact zero: the tokenizer gives an exact zero its own "
        "token, so any rounding noise flips it and the golden values are not reproducible across platforms")
    assert z.abs().min().item() > MIN_REAL_VALUE, (
        f"a standardised feature value is {z.abs().min().item():.1e}: the tokenizer's sensitivity there "
        "scales as 1/|z|, so the golden values would not be stable")
    s = y.shape[1]
    with torch.no_grad():
        noise = 0.0
        for k in range(3):
            g = torch.Generator().manual_seed(k)
            xp = x * (1 + 1e-6 * torch.randn(x.shape, generator=g))
            noise = max(noise, (wrapper(xp, y) - base).abs().max().item())
        y2 = y.clone()
        y2[0, 0] = (y2[0, 0] + 1) % 3 if task == "classification" else y2[0, 0] + 1.5
        label = (wrapper(x, y2)[0, s:] - base[0, s:]).abs().max().item()
    assert noise < MAX_NOISE_RESPONSE, (
        f"ill-conditioned fixture: a 1e-6 relative input change moves the output by {noise:.2e}; ORT and "
        "PyTorch would disagree by more than rounding and no golden value is stable")
    assert label > MIN_LABEL_SENSITIVITY, (
        f"input-blind fixture: changing a training label moves the query outputs by only {label:.2e}")


def assert_margins(logits: torch.Tensor) -> None:
    top2 = logits[0].topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).min().item()
    assert margin > MIN_MARGIN, (
        f"a golden row is a near-tie (min margin {margin:.3g}); the SQL test would be flaky across "
        "platforms. Change the seed or the input formulas.")


def safetensors_digest(model, path: pathlib.Path) -> str:
    sd = {k: v.detach().contiguous() for k, v in sorted(model.state_dict().items())}
    save_file(sd, str(path),
              metadata={"origin": "anofox-tabfm CI fixture, random init, Apache-2.0 LimiX "
                                  "architecture (our weights, not StableAI's)"})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _build_task(out: pathlib.Path, task: str, digest: str) -> None:
    cfg = x_configs.fixture()
    model = seeded_model()
    graph_path = out / f"graph_limix_{task}.onnx"
    wrapper = x_export.export_graph(model, graph_path, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                                    dim_features=cfg.dim_features, example=cfg.example,
                                    max_classes=cfg.max_classes, task=task)
    tensor_map = x_export.postprocess(graph_path, dict(model.state_dict()))
    x_export.write_tensor_map(out / f"tensor_map_limix_{task}.json", tensor_map, task=task,
                              safetensors_rel="model.safetensors")
    parity = x_export.check_parity(graph_path, wrapper, cfg.parity_shapes, cfg.max_classes, task=task,
                                   tol=1e-3)
    assert parity["ok"], parity
    x_export.delete_weight_data(graph_path)
    x_export.assert_weight_free(graph_path, tensor_map)

    x, y = golden_tensors(task)
    assert_well_conditioned(wrapper, model, x, y, task)
    with torch.no_grad():
        logits = wrapper(x, y)                                     # [1, T, C]
    s = GOLDEN_SHAPE["s"]
    per_row = {}
    if task == "classification":
        assert_margins(logits)
        per_row["labels"] = [f"c{int(c)}" for c in logits[0].argmax(-1).tolist()]
        assert len(set(per_row["labels"])) == 3, "the fixture must predict every class"
    else:
        values = logits[0, :, 0]
        assert values.std().item() > 0.05, "golden regression output is nearly constant"
        per_row["values"] = [round(v, 6) for v in values.tolist()]
    assert logits[0].std(dim=0).max().item() > 0.05, "golden output is nearly constant across rows"

    (out / f"golden_{task}.json").write_text(json.dumps({
        "_doc": {
            "purpose": f"Parity: safetensors -> initializer injection -> ORT on graph_limix_{task}.onnx must "
                       "reproduce these fp32 outputs, for EVERY row (rows < S are in-context fitted values, "
                       "rows >= S are predictions).",
            "rtol": PARITY_RTOL,
            "y_convention": "y holds ONLY the training targets (length S); S = len(y).",
            "inputs": "x[i][j] = ((i*7 + j*3 + (i*j) % 5) % 11) * 0.25 - 1.0 + "
                      "((i*13 + j*5 + (i*j) % 3) % 7) * 0.0625; "
                      + ("y[i] = (i*5 + i//3) % 3" if task == "classification"
                         else "y[i] = ((i*37) % 17) * 0.25 - 2.0"),
        },
        "inputs": {"x": x.tolist(), "y": y.tolist(), "train_size": s},
        "outputs": logits.tolist(),
        "output_shape": list(logits.shape),
        "per_row": per_row,
        "safetensors_sha256": digest,
    }, indent=2) + "\n")


def build(out: pathlib.Path) -> dict:
    """Build the dual-task committed CI fixture (both tasks + combined manifest)."""
    out.mkdir(parents=True, exist_ok=True)
    model, model2 = seeded_model(), seeded_model()
    sd, sd2 = model.state_dict(), model2.state_dict()
    assert sorted(sd) == sorted(sd2)
    for k in sd:
        assert torch.equal(sd[k], sd2[k]), f"nondeterministic weight {k}"
    digest = safetensors_digest(model, out / "model.safetensors")
    recheck = out / "model.safetensors.recheck"
    digest2 = safetensors_digest(model2, recheck)
    recheck.unlink()
    assert digest == digest2, f"safetensors bytes not deterministic: {digest} != {digest2}"

    for t in TASKS:
        _build_task(out, t, digest)

    def file_entry(name):
        p = out / name
        return {"path": name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}

    manifest = {
        "schema_version": 2, "id": "limix-fixture",
        "display_name": "LimiX CI fixture (random init, schema v2)",
        "family": "icl-transformer",
        "license": {"id": "apache-2.0", "commercial": True, "redistributable": True, "gate_setting": None},
        "preprocessing_profile": "limix_v1_raw",
        "weights": {
            t: {"repo": "local:test/fixtures/limix", "revision": "fixture-v1",
                "files": [file_entry("model.safetensors")]}
            for t in TASKS
        },
        "graph": {"classification": "graph_limix_classification.onnx",
                  "regression": "graph_limix_regression.onnx",
                  "tensor_map": {"classification": "tensor_map_limix_classification.json",
                                 "regression": "tensor_map_limix_regression.json"}},
        "capabilities": ["classify", "regress"],
        "tensor_contract": {
            "inputs": {
                "features": {"name": "x", "dtype": "f32", "shape": ["1", "T", "H"]},
                "labels": {"name": "y", "dtype": "f32", "shape": ["1", "S"]},
            },
            "outputs": {"logits": {"name": "logits", "dtype": "f32", "shape": ["1", "T", "C"]}},
        },
        "size_regime": {"max_rows": 4096, "max_features": 64, "max_classes": 3},
        "compute": {"cpu": "f32"},
        "_note": "Random-init LimiX fixture (Apache-2.0 architecture, our weights). ONE weights file serves "
                 "both tasks. H is DYNAMIC; y carries train targets only (length = train_size); no "
                 "cat_mask/d/train_size inputs. The model standardises features itself and the wrapper "
                 "standardises the regression target in-graph, so the engine profile is *_raw. "
                 "classification outputs are class logits (C = max_classes); regression outputs are a "
                 "SINGLE RAW-space point estimate (C = 1). Rows < S are in-context fitted values "
                 "(context rows presented a second time as queries).",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    hashes = {name: sha256_file(out / name) for name in FILES}
    total = sum((out / name).stat().st_size for name in FILES)
    assert total < 5 * 1024 * 1024, f"fixture total {total} B >= 5 MB budget"
    return hashes
