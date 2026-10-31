"""The transcribed configs must equal the checkpoints' own `config` blocks.

Skipped when the checkpoints are not downloaded (CI never fetches them). Every
architecture field the checkpoint carries must match; keys only the checkpoint
has are runtime knobs (`icl_bf16`, `name`, the use_* flags that equal the
defaults) and are compared too when the dataclass knows them.
"""

import json
import pathlib
import struct

import pytest

import tabpfn.architectures.tabpfn_v3_5 as v35

from export_tabpfn35 import configs

CACHE = pathlib.Path.home() / ".cache/anofox-tabfm/Prior-Labs__tabpfn_3_5@main"


def _ckpt_config(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(json.loads(f.read(n))["__metadata__"]["config"])


@pytest.mark.parametrize("variant,kwargs", [("v3.5", configs.REAL35), ("v3.5-fast", configs.REAL35_FAST)])
def test_transcription_matches_checkpoint(variant, kwargs):
    path = CACHE / variant / "model.safetensors"
    if not path.exists():
        pytest.skip(f"{path} not downloaded")
    ck = _ckpt_config(path)
    ours = v35.TabPFNV3p5Config(**kwargs)
    theirs = v35.TabPFNV3p5Config(**{k: v for k, v in ck.items()
                                     if k in v35.TabPFNV3p5Config.__dataclass_fields__})
    assert ours == theirs
