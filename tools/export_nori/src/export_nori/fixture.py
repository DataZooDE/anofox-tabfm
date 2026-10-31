"""CI fixture model for Nori-30M: random weights, tiny dims, deterministic bytes.

Same shape as the LimiX-2M fixture (Nori is its fork): a tiny random-init Nori (Apache-2.0 architecture, OUR
weights, zero Synthefy checkpoint bytes: see CLAUDE.md "license wall") at every structural flag of the
released config, a weight-free graph through the same exporter, and a v2 manifest with the ONE task the
model has. The weights file is a ``torch.save`` checkpoint in the released file's own layout
(``{"model_config", "ema_state_dict"}``), because that is what the built-in entry hands the engine: the
fixture therefore exercises the native checkpoint reader on the Nori layout, not a safetensors stand-in.

What carries over from LimiX (the tokenizer is the same scientific-notation RBF):

  * an EXACT standardised zero is its own token and values near zero are hypersensitive in proportion to
    1/|z|, so the golden inputs use LimiX's quarter + sixteenths formula and ``assert_well_conditioned``
    rejects a fixture with a standardised value closer to zero than MIN_REAL_VALUE;
  * a default initialisation is nearly input-blind, so the target-side projections are scaled up
    (TARGET_GAIN) until the output depends on the training targets; the seed is the first that passes
    every check.

What is new for Nori: upstream initialises every attention output projection and SwiGLU down projection
at std 0.02 and the DeepNorm residual scale is 0.19, so a random 2-layer stack barely moves the target
token: every query row decoded to the same value (output std ~1e-4 over 20 rows, for 40 seeds in a row).
Scaling those two projections (MIX_GAIN) lets the sample and feature attention carry the table into the
target token, which is what the golden values must exercise. Second, the point estimate is the MEAN of
the quantile head, and K independent random head rows average each other out; the head is therefore
given one shared direction (HEAD_GAIN) plus an ordering ramp, as a trained head's quantiles share a
location. Only the context label embedding is target-gained: gaining the label ENCODER makes the masked
query-row label token so large that the final RMSNorm flattens every query row to the same value.

The golden inputs are exact in binary floating point and expressible in SQL, so the SQL test builds the
same table and pins a prediction for EVERY row, context and query.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import torch

from export_nori import configs as x_configs
from export_nori import export as x_export
from export_nori.nori_patches import ExportWrapper, build_model

SEED_WEIGHTS = 0
TARGET_GAIN = 10.0
MIX_GAIN = 50.0                            # attention output / SwiGLU down projections; see the docstring
HEAD_GAIN = 20.0                           # the quantile head's shared direction; see sensitive()
MIN_TARGET_SENSITIVITY = 0.1               # query output change when one training target changes by 1.5
MAX_NOISE_RESPONSE = 1e-2                  # output change per 1e-6 relative input perturbation
MIN_REAL_VALUE = 0.02                      # smallest |standardised feature value| (tokenizer 1/|z| exposure)
MIN_SPREAD = 0.05                          # the golden output must not be nearly constant
GOLDEN_SHAPE = dict(t=20, h=5, s=12)       # T rows, H features, S train rows
TASK = "regression"
WEIGHTS = "model.ckpt"
FILES = ["graph_nori_regression.onnx", "tensor_map_nori_regression.json", "golden_regression.json",
         WEIGHTS, "manifest.json"]


def sql_inputs():
    """(x [T,H], y [S]) as nested lists, from formulas the SQL test repeats verbatim.

    x[i][j] = ((i*7 + j*3 + (i*j) % 5) % 11) * 0.25 - 1.0 + ((i*13 + j*5 + (i*j) % 3) % 7) * 0.0625
    y[i]    = ((i*37) % 17) * 0.25 - 2.0           exact in float32
    """
    t, h, s = (GOLDEN_SHAPE[k] for k in ("t", "h", "s"))
    x = [[((i * 7 + j * 3 + (i * j) % 5) % 11) * 0.25 - 1.0 + ((i * 13 + j * 5 + (i * j) % 3) % 7) * 0.0625
          for j in range(h)] for i in range(t)]
    y = [((i * 37) % 17) * 0.25 - 2.0 for i in range(s)]
    return x, y


def golden_tensors():
    xs, ys = sql_inputs()
    return torch.tensor([xs], dtype=torch.float32), torch.tensor([ys], dtype=torch.float32)


def sensitive(model):
    """Scale ONLY the target-side projections of a default-initialised model (see the docstring)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim < 2:
                continue
            if "reg_target_aware_embedding" in name:
                param.mul_(TARGET_GAIN)
            elif name.endswith(("out_proj_weight", "w_down.weight")):
                param.mul_(MIX_GAIN)
        # The quantile head's K independent random rows average out in the point estimate (its mean over
        # K quantiles), which leaves the golden output flat whatever the encoder does. A trained head's
        # quantiles are strongly correlated (they bracket one location), so give every row one shared
        # direction plus a per-quantile offset that orders them: W_k = W_0 + (k - K/2) / K * W_1.
        head = model.reg_y_decoder[-1].weight
        k = head.shape[0]
        ramp = (torch.arange(k, dtype=head.dtype) - k / 2).unsqueeze(1) / k
        head.copy_(HEAD_GAIN * (head[:1] + ramp * head[1:2]))
    return model


def seeded_model(seed: int = SEED_WEIGHTS):
    # build_model seeds torch before constructing, so two calls give identical weights.
    return sensitive(build_model(x_configs.fixture().model_config, seed=seed)).eval()


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


def check_well_conditioned(wrapper, model, x, y) -> list:
    """Every reason the fixture is unusable (empty when it is fine): it must respond to its targets, must
    NOT amplify rounding noise, and must not contain a feature value the tokenizer is singular at."""
    problems = []
    z, base = _real_standardised(model, wrapper, x, y)
    zeros = int((z == 0).sum())
    if zeros:
        problems.append(f"{zeros} standardised feature value(s) are an exact zero (the tokenizer's own token)")
    if z.abs().min().item() <= MIN_REAL_VALUE:
        problems.append(f"a standardised feature value is {z.abs().min().item():.1e} (1/|z| sensitivity)")
    s = y.shape[1]
    with torch.no_grad():
        noise = 0.0
        for k in range(3):
            g = torch.Generator().manual_seed(k)
            xp = x * (1 + 1e-6 * torch.randn(x.shape, generator=g))
            noise = max(noise, (wrapper(xp, y) - base).abs().max().item())
        y2 = y.clone()
        y2[0, 0] = y2[0, 0] + 1.5
        sens = (wrapper(x, y2)[0, s:] - base[0, s:]).abs().max().item()
    if noise >= MAX_NOISE_RESPONSE:
        problems.append(f"ill-conditioned: a 1e-6 relative input change moves the output by {noise:.2e}")
    if sens <= MIN_TARGET_SENSITIVITY:
        problems.append(f"target-blind: changing one training target moves the queries by {sens:.2e}")
    values = base[0, :, 0]
    if values.std().item() <= MIN_SPREAD:
        problems.append(f"nearly constant output (std {values.std().item():.3g})")
    if values[:s].std().item() <= MIN_SPREAD:
        problems.append("the context (fitted) rows are nearly constant")
    return problems


def assert_well_conditioned(wrapper, model, x, y) -> None:
    problems = check_well_conditioned(wrapper, model, x, y)
    assert not problems, "; ".join(problems)


def save_checkpoint(model, path: pathlib.Path) -> str:
    """The released file's layout: {"model_config": <the fixture config>, "ema_state_dict": <weights>}.
    torch.save output is deterministic for the same tensors and config (checked by ``build``)."""
    sd = {k: v.detach().contiguous().clone() for k, v in sorted(model.state_dict().items())}
    torch.save({"model_config": x_configs.fixture().model_config, "ema_state_dict": sd}, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(out: pathlib.Path) -> dict:
    """Build the committed CI fixture (regression graph + checkpoint + golden + manifest)."""
    out.mkdir(parents=True, exist_ok=True)
    model, model2 = seeded_model(), seeded_model()
    for k, v in model.state_dict().items():
        assert torch.equal(v, model2.state_dict()[k]), f"nondeterministic weight {k}"
    # torch.save names the archive root after the file stem, so the recheck must have the same stem.
    digest = save_checkpoint(model, out / WEIGHTS)
    recheck_dir = out / ".recheck"
    recheck_dir.mkdir(exist_ok=True)
    digest2 = save_checkpoint(model2, recheck_dir / WEIGHTS)
    (recheck_dir / WEIGHTS).unlink()
    recheck_dir.rmdir()
    assert digest == digest2, f"checkpoint bytes not deterministic: {digest} != {digest2}"

    cfg = x_configs.fixture()
    graph_path = out / f"graph_nori_{TASK}.onnx"
    wrapper = x_export.export_graph(model, graph_path, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                                    dim_features=cfg.dim_features, example=cfg.example)
    tensor_map = x_export.postprocess(graph_path, dict(model.state_dict()))
    x_export.write_tensor_map(out / f"tensor_map_nori_{TASK}.json", tensor_map, task=TASK,
                              safetensors_rel=WEIGHTS)
    parity = x_export.check_parity(graph_path, wrapper, cfg.parity_shapes, tol=1e-3)
    assert parity["ok"], parity
    x_export.delete_weight_data(graph_path)
    x_export.assert_weight_free(graph_path, tensor_map)

    x, y = golden_tensors()
    assert_well_conditioned(wrapper, model, x, y)
    with torch.no_grad():
        out_t = wrapper(x, y)                                         # [1, T, 1]
    s = GOLDEN_SHAPE["s"]
    values = [round(v, 6) for v in out_t[0, :, 0].tolist()]
    (out / f"golden_{TASK}.json").write_text(json.dumps({
        "_doc": {
            "purpose": "Parity: checkpoint -> initializer injection -> ORT on graph_nori_regression.onnx must "
                       "reproduce these fp32 outputs, for EVERY row (rows < S are in-context fitted values, "
                       "rows >= S are predictions).",
            "y_convention": "y holds ONLY the RAW training targets (length S); S = len(y).",
            "inputs": "x[i][j] = ((i*7 + j*3 + (i*j) % 5) % 11) * 0.25 - 1.0 + "
                      "((i*13 + j*5 + (i*j) % 3) % 7) * 0.0625; y[i] = ((i*37) % 17) * 0.25 - 2.0",
        },
        "inputs": {"x": x.tolist(), "y": y.tolist(), "train_size": s},
        "outputs": out_t.tolist(),
        "output_shape": list(out_t.shape),
        "per_row": {"values": values},
        "checkpoint_sha256": digest,
    }, indent=2) + "\n")

    def file_entry(name):
        p = out / name
        return {"path": name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}

    manifest = {
        "schema_version": 2, "id": "nori-fixture",
        "display_name": "Nori CI fixture (random init, schema v2)",
        "family": "icl-transformer",
        "license": {"id": "apache-2.0", "commercial": True, "redistributable": True, "gate_setting": None},
        "preprocessing_profile": "nori_v1_raw",
        "weights": {TASK: {"repo": "local:test/fixtures/nori", "revision": "fixture-v1",
                           "files": [file_entry(WEIGHTS)]}},
        "graph": {TASK: f"graph_nori_{TASK}.onnx", "tensor_map": {TASK: f"tensor_map_nori_{TASK}.json"}},
        "capabilities": ["regress"],
        "tensor_contract": {
            "inputs": {
                "features": {"name": "x", "dtype": "f32", "shape": ["1", "T", "H"]},
                "labels": {"name": "y", "dtype": "f32", "shape": ["1", "S"]},
            },
            "outputs": {"logits": {"name": "logits", "dtype": "f32", "shape": ["1", "T", "1"]}},
        },
        "size_regime": {"max_rows": 4096, "max_features": 64, "max_classes": 10},
        "compute": {"cpu": "f32"},
        "_note": "Random-init Nori fixture (Apache-2.0 architecture, our weights), regression only, in the "
                 "released checkpoint's own torch.save layout {model_config, ema_state_dict}. H is DYNAMIC; y "
                 "carries RAW train targets only (length = train_size). The model standardises features "
                 "itself and the wrapper standardises the target in-graph, so the engine profile is *_raw. "
                 "The output is ONE raw-space point estimate per row (the mean of the quantile head); rows < "
                 "S are in-context fitted values (context rows presented a second time as queries).",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    hashes = {name: sha256_file(out / name) for name in FILES}
    total = sum((out / name).stat().st_size for name in FILES)
    assert total < 5 * 1024 * 1024, f"fixture total {total} B >= 5 MB budget"
    return hashes
