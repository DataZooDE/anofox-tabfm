# MLX spikes (docs/MLX_PLAN.md) — run on the Apple-Silicon Mac

Each spike is one self-contained script that prints `MARKER:` lines; the
verdict lives in those lines, not in "it seemed to work". Run from this
directory:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install mlx numpy onnx onnxruntime safetensors huggingface_hub
python s_m1_mlx_onnx.py      # feasibility: can mlx-onnx run our ext graph?
python s_m2_handport.py      # parity: hand-ported mitra forward vs golden
```

`golden_mitra_classification.json` was generated on the Linux box by
`make_golden.py` from ORT CPU + the real mitra weights — the SAME inputs are
re-derived deterministically on the Mac, so parity is a straight comparison.
