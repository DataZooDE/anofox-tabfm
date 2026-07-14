"""Download the real TabICL v2 checkpoint (.ckpt) and write a safetensors the
extension can inject, keyed by the committed tensor map. One-time dev-side step
(the extension stays pure C++/ORT). No weights committed.

Usage:  uv run python convert_weights.py <task> <cache_dir>
"""
import json, pathlib, sys
import torch
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download

task = sys.argv[1] if len(sys.argv) > 1 else "classification"
cache_dir = pathlib.Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else pathlib.Path.home() / ".cache/anofox-tabfm"
fname = {"classification": "tabicl-classifier-v2-20260212.ckpt",
         "regression": "tabicl-regressor-v2-20260212.ckpt"}[task]

tmap_path = pathlib.Path(__file__).resolve().parents[2] / f"resources/tensor_map_tabicl_{task}.json"
tmap = json.load(open(tmap_path))
inits = tmap.get("initializers", tmap)
want = set(inits.values())

print(f"downloading {fname} from jingang/TabICL ...", flush=True)
ckpt_path = hf_hub_download(repo_id="jingang/TabICL", filename=fname)
obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
# ckpt may be a raw state_dict or wrap one under a key
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
if missing[:5]:
    print("  sample missing:", missing[:5])
    print("  sample ckpt keys:", list(sd)[:5])
if not present:
    sys.exit(2)
tensors = {k: sd[k].detach().to(torch.float32).contiguous() for k in present}
out = cache_dir / f"jingang__TabICL@main" / task / "model.safetensors"
out.parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(out))
print(f"wrote {len(tensors)} tensors -> {out} ({out.stat().st_size} bytes)")
