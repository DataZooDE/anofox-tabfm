#!/usr/bin/env python3
# Rewrite the weight-free graph so every initializer references the cached
# safetensors file as ONNX external-data (location/offset/length). The
# safetensors data section is raw, contiguous, row-major tensor bytes, so the
# ONNX external offset is simply 8 + header_len + st_data_offset.
import json, os, struct, sys
import onnx
from onnx import TensorProto

GRAPH = "resources/graph_classification.onnx"
TMAP = "resources/tensor_map_classification.json"
ST = os.path.expanduser(
    "~/.cache/anofox-tabfm/google__tabfm-1.0.0-pytorch@main/classification/model.safetensors")
OUT_DIR = os.path.dirname(ST)
OUT = os.path.join(OUT_DIR, "graph_ext.onnx")

with open(ST, "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hlen))
data_base = 8 + hlen
st_off = {k: v["data_offsets"] for k, v in header.items() if k != "__metadata__"}

with open(TMAP) as f:
    onnx2st = json.load(f)["initializers"]

m = onnx.load(GRAPH, load_external_data=False)
n_ext = 0
missing = []
for init in m.graph.initializer:
    st_key = onnx2st.get(init.name)
    if st_key is None:
        # engine fallback: strip a leading "m."
        st_key = init.name[2:] if init.name.startswith("m.") else init.name
    if st_key not in st_off:
        missing.append(init.name)
        continue
    b, e = st_off[st_key]
    abs_off, length = data_base + b, e - b
    # clear any inline/stub data, mark EXTERNAL with real location+offset+length
    init.ClearField("raw_data")
    for fld in ("float_data", "int32_data", "int64_data", "double_data"):
        init.ClearField(fld)
    init.data_location = TensorProto.EXTERNAL
    del init.external_data[:]
    for k, v in (("location", "model.safetensors"), ("offset", str(abs_off)), ("length", str(length))):
        entry = init.external_data.add()
        entry.key, entry.value = k, v
    n_ext += 1

print(f"initializers: {len(m.graph.initializer)}  externalized: {n_ext}  missing: {len(missing)}")
if missing:
    print("  MISSING (first 5):", missing[:5])
onnx.save(m, OUT)  # keep external refs as-is (do NOT re-embed)
print("wrote", OUT, os.path.getsize(OUT), "bytes  (proto stays tiny — weights stay in the safetensors)")
