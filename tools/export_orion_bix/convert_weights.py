"""Download the real Orion-BiX checkpoint (.ckpt) and write a safetensors the
extension can inject, keyed by the committed tensor map. One-time dev-side step
(the extension stays pure C++/ORT). No weights committed.

Note the extension does NOT need this at runtime: `tabfm_download` fetches the
.ckpt straight from HuggingFace and `src/tabfm_ckpt.cpp` reads the torch zip
natively, unwrapping the `state_dict` key. This script exists to *verify* the
committed tensor map covers the real checkpoint 1:1, and to stage a safetensors
copy for offline parity runs.

Usage:  uv run python convert_weights.py [cache_dir]
"""
import json, pathlib, sys
import torch
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download

TASK = "classification"  # Orion-BiX ships no regressor
REPO = "Lexsi/Orion-BiX"
FNAME = "Orion-BiX-v1.1.ckpt"

cache_dir = pathlib.Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 \
    else pathlib.Path.home() / ".cache/anofox-tabfm"

tmap_path = pathlib.Path(__file__).resolve().parents[2] / f"resources/tensor_map_orion_bix_{TASK}.json"
tmap = json.load(open(tmap_path))
inits = tmap.get("initializers", tmap)
want = set(inits.values())

print(f"downloading {FNAME} from {REPO} ...", flush=True)
ckpt_path = hf_hub_download(repo_id=REPO, filename=FNAME)
obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
# Orion-BiX saves {"state_dict":..., "config":..., "optimizer_state":...}
sd = obj
for key in ("state_dict", "model", "model_state_dict", "weights"):
    if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
        sd = obj[key]
        break
print("ckpt top-level type:", type(obj).__name__,
      "| state_dict tensors:", sum(1 for v in sd.values() if torch.is_tensor(v)))

present = [k for k in want if k in sd]
missing = [k for k in want if k not in sd]
print(f"tensor-map keys: {len(want)} | present in ckpt: {len(present)} | missing: {len(missing)}")
if missing:
    print("  sample missing:", missing[:5])
    print("  sample ckpt keys:", list(sd)[:5])
    # A partial map means the committed graph does not match the released
    # architecture — injection would leave initializers unbound at runtime.
    sys.exit(2)

tensors = {k: sd[k].detach().to(torch.float32).contiguous() for k in present}
out = cache_dir / "Lexsi__Orion-BiX@main" / TASK / "model.safetensors"
out.parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(out))
print(f"wrote {len(tensors)} tensors -> {out} ({out.stat().st_size} bytes)")
