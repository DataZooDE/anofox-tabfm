"""Download a real TabPFN checkpoint (.ckpt pickle) and write a safetensors the
extension can inject, keyed by the committed tensor map's checkpoint keys.
Places it at the extension cache slug so `model := 'tabpfn-v2'` (or
`'tabpfn-v2-5'`) runs with real weights. No weights are committed; this is a
user-side one-time conversion (the extension stays pure C++/ORT — dev tooling,
run once).

Usage:  uv run python convert_weights.py <task> [cache_dir] [--arch v2|v2.5]
        task = classification | regression
"""
import json, pathlib, sys, tempfile
import torch
from safetensors.torch import save_file
from export_tabpfn.tabpfn_patched import load_real_model

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
arch = "v2"
for a in sys.argv[1:]:
    if a.startswith("--arch"):
        arch = a.split("=", 1)[1] if "=" in a else "v2.5"

task = argv[0] if argv else "classification"
cache_dir = pathlib.Path(argv[1]).expanduser() if len(argv) > 1 \
    else pathlib.Path.home() / ".cache/anofox-tabfm"
which = {"classification": "classifier", "regression": "regressor"}[task]

if arch == "v2.5":
    slug_name = "tabpfn25"
    repo_slug = "Prior-Labs__tabpfn_2_5"
    hf_file = f"tabpfn-v2.5-{which}-v2.5_default.ckpt"
    hf_url = f"https://huggingface.co/Prior-Labs/tabpfn_2_5/resolve/main/{hf_file}"
else:
    slug_name = "tabpfn"
    repo_slug = {"classification": "Prior-Labs__TabPFN-v2-clf",
                 "regression": "Prior-Labs__TabPFN-v2-reg"}[task]

tmap_path = (pathlib.Path(__file__).resolve().parents[2]
             / f"resources/tensor_map_{slug_name}_{task}.json")
tmap = json.load(open(tmap_path))
inits = tmap.get("initializers", tmap)  # onnx-init-name -> checkpoint key
want_keys = set(inits.values())  # checkpoint-namespace keys the graph references

tmp = pathlib.Path(tempfile.mkdtemp())
if arch == "v2.5":
    # The tabpfn_2_5 repo carries `extra_gated_fields`, but the resolve endpoint
    # serves anonymously; if that ever changes this download 401s and the user
    # needs an HF token (see docs/REAL_MODELS.md).
    import urllib.request
    ckpt = tmp / hf_file
    print(f"downloading {hf_file} (transient) ...", flush=True)
    urllib.request.urlretrieve(hf_url, ckpt)
else:
    from tabpfn.model_loading import download_model, ModelVersion
    ckpt = tmp / f"tabpfn-v2-{which}.ckpt"
    print(f"downloading real TabPFN v2 {which} ckpt (transient) ...", flush=True)
    download_model(to=ckpt, version=ModelVersion.V2, which=which, model_name=ckpt.name)
    ckpt = next(tmp.rglob("*.ckpt"))

model = load_real_model(task, str(ckpt), arch=arch)
sd = model.state_dict()
tensors, missing = {}, []
for k in sorted(want_keys):
    if k in sd:
        tensors[k] = sd[k].detach().to(torch.float32).contiguous()
    else:
        missing.append(k)
print(f"tensor-map keys: {len(want_keys)} | present: {len(tensors)} | missing: {len(missing)}",
      flush=True)
if missing:
    print(f"WARNING: {len(missing)} keys absent from state_dict (first: {missing[:3]})",
          flush=True)

out = cache_dir / f"{repo_slug}@main" / task / "model.safetensors"
out.parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(out))
print(f"wrote {len(tensors)} tensors -> {out} ({out.stat().st_size} bytes)", flush=True)
import shutil; shutil.rmtree(tmp, ignore_errors=True)
