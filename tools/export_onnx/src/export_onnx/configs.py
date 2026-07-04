"""Export configurations: tiny / fixture / real.

`tiny` and `fixture` use the TabFM.__init__ defaults (embed_dim=8,
max_classes=3, ~17k params). `real` reads the architecture dims from the real
model's config.json, fetched anonymously from Hugging Face — dims only, the
model is then built with RANDOM weights (the exported graph is
architecture-only and weight-free, so no Google weight bytes are involved at
any point; BRD FR-2.2 / HLD license wall).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import urllib.request

HF_CONFIG_URL = (
    "https://huggingface.co/google/tabfm-1.0.0-pytorch/resolve/main/"
    "{task}/config.json"
)

# S01 shipping dynamic ranges.
DIM_ROWS = ("rows", 4, 100_000)
DIM_FEATURES = ("features", 2, 512)
# S06 fixture ranges (unit-test sized).
DIM_ROWS_FIXTURE = ("rows", 4, 4096)
DIM_FEATURES_FIXTURE = ("features", 2, 64)

OPSET = 18


@dataclasses.dataclass(frozen=True)
class ExportConfig:
  name: str
  model_kwargs: dict  # TabFM(**model_kwargs); is_classifier set per task
  dim_rows: tuple  # (name, min, max) for torch.export.Dim
  dim_features: tuple
  # export example input shape (T, H, d, train). Parity shapes must differ.
  example: tuple
  parity_shapes: tuple  # ((T, H, d, train), ...)
  config_source: dict | None = None  # provenance for `real`

  @property
  def max_classes(self) -> int:
    return self.model_kwargs.get("max_classes", 3)


# TabFM.__init__ defaults, spelled out (S06 FIXTURE_CFG minus is_classifier).
_TINY_KWARGS = dict(
    embed_dim=8, max_classes=3, col_num_blocks=2, col_nhead=2, col_num_inds=4,
    row_num_blocks=2, row_nhead=2, row_num_cls=2, icl_num_blocks=2,
    icl_nhead=2, ff_factor=2, feature_group_size=3, num_freq=32,
)

# S01 DoD parity shapes — all different from the export example (T=64, H=12).
_S01_PARITY = ((16, 4, 4, 12), (300, 25, 20, 240), (1200, 80, 77, 1000))


def tiny() -> ExportConfig:
  return ExportConfig(
      name="tiny",
      model_kwargs=dict(_TINY_KWARGS),
      dim_rows=DIM_ROWS,
      dim_features=DIM_FEATURES,
      example=(64, 12, 10, 48),
      parity_shapes=_S01_PARITY,
  )


def fixture() -> ExportConfig:
  # Same dims as tiny; export example matches the S06 golden shape and the
  # dynamic ranges are the S06 unit-test ranges. Parity shape differs from
  # the export example (S06: T=40, H=7, d=6, train=30).
  return ExportConfig(
      name="fixture",
      model_kwargs=dict(_TINY_KWARGS),
      dim_rows=DIM_ROWS_FIXTURE,
      dim_features=DIM_FEATURES_FIXTURE,
      example=(12, 5, 5, 8),
      parity_shapes=((40, 7, 6, 30),),
  )


def fetch_real_config(task: str, config_json_path: str | None = None) -> dict:
  """Read the real model's config.json (anonymous HTTPS or a local file).

  Returns {"kwargs": <TabFM kwargs>, "source": {url/path, sha256}}.
  Only architecture dims travel; never weights.
  """
  if config_json_path is not None:
    raw = open(config_json_path, "rb").read()
    origin = str(config_json_path)
  else:
    url = HF_CONFIG_URL.format(task=task)
    with urllib.request.urlopen(url, timeout=60) as resp:
      raw = resp.read()
    origin = url
  cfg = json.loads(raw)
  sha = hashlib.sha256(raw).hexdigest()
  # config.json holds TabFM kwargs directly (plus is_classifier, which the
  # caller overrides per task; decoder_hidden may be null).
  kwargs = {k: v for k, v in cfg.items() if k != "is_classifier"}
  if kwargs.get("decoder_hidden") is None:
    kwargs.pop("decoder_hidden", None)
  return {"kwargs": kwargs, "source": {"origin": origin, "sha256": sha},
          "raw": cfg}


def real(task: str, config_json_path: str | None = None) -> ExportConfig:
  fetched = fetch_real_config(task, config_json_path)
  return ExportConfig(
      name="real",
      model_kwargs=fetched["kwargs"],
      dim_rows=DIM_ROWS,
      dim_features=DIM_FEATURES,
      example=(64, 12, 10, 48),
      # One small shape by default — a 1.6B fp32 random-weight forward per
      # framework is minutes on CPU; still different from the export example.
      parity_shapes=((16, 4, 4, 12),),
      config_source=fetched["source"],
  )


def get(name: str, task: str, config_json_path: str | None = None) -> ExportConfig:
  if name == "tiny":
    return tiny()
  if name == "fixture":
    return fixture()
  if name == "real":
    return real(task, config_json_path)
  raise ValueError(f"unknown config {name!r} (tiny|fixture|real)")
