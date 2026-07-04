"""anofox-tabfm S01 exporter package.

Import `export_onnx.export` for the pipeline pieces; `export_onnx.cli` is the
`uv run export_onnx` entry point. The TabFM model class comes from
`export_onnx.tabfm_model_patched` — a patched copy of the vendor model
(2-line repeat_interleave rewrite, see that file's header). The vendor
package itself is never imported: its __init__ drags sklearn/pandas/jax.
"""
