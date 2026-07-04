"""Standalone import of vendor/tabfm's classifier_and_regressor.py.

The vendor package __init__ drags jax/flax; the module itself only needs
numpy/pandas/scipy/sklearn/absl/jaxtyping/typeguard (jax + torch are
optional imports inside it). We load the FILE directly so the golden
vectors come from the exact upstream code, unmodified.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[4]
VENDOR_CAR = REPO / "vendor/tabfm/tabfm/src/classifier_and_regressor.py"

_MOD_NAME = "tabfm_classifier_and_regressor_standalone"


def load():
  if _MOD_NAME in sys.modules:
    return sys.modules[_MOD_NAME]
  spec = importlib.util.spec_from_file_location(_MOD_NAME, VENDOR_CAR)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[_MOD_NAME] = mod
  spec.loader.exec_module(mod)
  return mod
