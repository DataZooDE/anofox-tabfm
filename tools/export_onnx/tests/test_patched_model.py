"""The patched model copy must be the vendor file plus EXACTLY the S01 diff."""

import pathlib

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO = TOOLS_DIR.parents[1]
VENDOR = REPO / "vendor/tabfm/tabfm/src/pytorch/model.py"
PATCHED = TOOLS_DIR / "src/export_onnx/tabfm_model_patched.py"

# The exact 2-line rewrite from S01 RESULTS.md ("model.py patch").
DIFF = [
    (
        "    ts = train_size.repeat_interleave(hc)  # [B*HC]\n",
        "    ts = train_size[:, None].expand(b, hc).reshape(b * hc)  # [B*HC]\n",
    ),
    (
        "      mask = valid.repeat_interleave(t, dim=0)[:, None, None, :]"
        "  # [B*T, 1, 1, HC]\n",
        "      mask = (valid[:, None, :].expand(b, t, hc)\n"
        "              .reshape(b * t, hc))[:, None, None, :]"
        "  # [B*T, 1, 1, HC]\n",
    ),
]


def test_patched_copy_is_vendor_plus_exact_diff():
  vendor_src = VENDOR.read_text()
  patched_src = PATCHED.read_text()

  expected = vendor_src
  for old, new in DIFF:
    assert old in expected, f"vendor drifted: expected line missing: {old!r}"
    assert expected.count(old) == 1
    expected = expected.replace(old, new)

  # Allow a provenance header on the patched copy, nothing else.
  header_end = patched_src.index("# Copyright 2026 Google LLC")
  header = patched_src[:header_end]
  assert "PATCHED COPY" in header
  assert patched_src[header_end:] == expected, (
      "tabfm_model_patched.py differs from vendor model.py beyond the "
      "documented 2-line repeat_interleave rewrite")


def test_no_repeat_interleave_on_dynamic_dims_left():
  src = PATCHED.read_text()
  # rope cos/sin repeat_interleave(2, -1) over the static head dim is fine;
  # the two dynamic-dim call sites must be gone.
  assert "train_size.repeat_interleave" not in src
  assert "valid.repeat_interleave" not in src
