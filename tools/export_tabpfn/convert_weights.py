"""Download the real TabPFN v2 checkpoint (.ckpt pickle) and write a safetensors
the extension can inject, keyed by the committed tensor map's checkpoint keys.
Places it at the extension cache slug so `model := 'tabpfn-v2'` runs with real
weights. No weights are committed; this is a user-side one-time conversion (the
extension stays pure C++/ORT — this is dev tooling, run once).

Usage:  uv run python convert_weights.py <task> <cache_dir>
        task = classification | regression
"""
import json, pathlib, sys, tempfile
import torch
from safetensors.torch import save_file
from export_tabpfn.tabpfn_patched import load_real_model

task = sys.argv[1] if len(sys.argv) > 1 else "classification"
cache_dir = pathlib.Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else pathlib.Path.home() / ".cache/anofox-tabfm"
repo_slug = {"classification": "Prior-Labs__TabPFN-v2-clf", "regression": "Prior-Labs__TabPFN-v2-reg"}[task]
which = {"classification": "classifier", "regression": "regressor"}[task]

tmap_path = pathlib.Path(__file__).resolve().parents[2] / f"resources/tensor_map_tabpfn_{task}.json"
tmap = json.load(open(tmap_path))
inits = tmap.get("initializers", tmap)  # onnx-init-name -> checkpoint key
want_keys = set(inits.values())  # checkpoint-namespace keys the graph references

from tabpfn.model_loading import download_model, ModelVersion
tmp = pathlib.Path(tempfile.mkdtemp())
ckpt = tmp / f"tabpfn-v2-{which}.ckpt"
print(f"downloading real TabPFN v2 {which} ckpt (transient) ...", flush=True)
download_model(to=ckpt, version=ModelVersion.V2, which=which, model_name=ckpt.name)
ckpt = next(tmp.rglob("*.ckpt"))

model = load_real_model(task, str(ckpt))
sd = model.state_dict()
tensors, missing = {}, []
for k in sorted(want_keys):
    if k in sd:
        tensors[k] = sd[k].detach().to(torch.float32).contiguous()
    else:
        missing.append(k)
if missing:
    print(f"WARNING: {len(missing)} tensor-map keys absent from state_dict (first: {missing[:3]})", flush=True)

out = cache_dir / f"{repo_slug}@main" / task / "model.safetensors"
out.parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(out))
print(f"wrote {len(tensors)} tensors -> {out} ({out.stat().st_size} bytes)", flush=True)
import shutil; shutil.rmtree(tmp, ignore_errors=True)
