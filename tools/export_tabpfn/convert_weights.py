"""Download a real TabPFN checkpoint (.ckpt pickle) and write a safetensors the
extension can inject, keyed by the committed tensor map's checkpoint keys.
Places it at the extension cache slug so `model := 'tabpfn-v2'` (or
`'tabpfn-v2-5'`, `'tabpfn-v2-5-real'`, `'tabpfn-v2-6'`, `'tabpfn-v3'`) runs with
real weights. No weights are committed; this is a user-side one-time conversion
(the extension stays pure C++/ORT — dev tooling, run once).

Usage:  uv run python convert_weights.py <task> [cache_dir]
                [--arch v2|v2.5|v2.6|v3] [--variant default|real]

        task    = classification | regression
        variant = which checkpoint of that generation to convert. `real` selects
                  the real-data continued pre-training (RealTabPFN-2.5, registry
                  id `tabpfn-v2-5-real`) and is 2.5-only.

The cache slug is keyed by HF REPO, and RealTabPFN-2.5 lives in the SAME repo as
TabPFN-2.5 (Prior-Labs/tabpfn_2_5). Writing both under `<task>/` would make the
two registry entries silently overwrite each other's weights, so the `real`
variant gets its own `<task>-real/` subdirectory — matching the files[].path the
built-in manifest declares in src/tabfm_registry.cpp.
"""
import json, pathlib, sys, tempfile
import torch
from safetensors.torch import save_file
from export_tabpfn.tabpfn_patched import load_real_model

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
arch = "v2"
variant = "default"
for a in sys.argv[1:]:
    if a.startswith("--arch"):
        arch = a.split("=", 1)[1] if "=" in a else "v2.5"
    elif a.startswith("--variant"):
        variant = a.split("=", 1)[1] if "=" in a else "real"
if arch not in ("v2", "v2.5", "v2.6", "v3"):
    # Previously any unknown --arch silently fell through to the v2 branch and
    # rewrote the v2 safetensors under the name of whatever was asked for,
    # reporting success. Refuse instead.
    raise SystemExit(f"unknown --arch {arch!r}; expected v2, v2.5, v2.6 or v3")
if variant not in ("default", "real"):
    raise SystemExit(f"unknown --variant {variant!r}; expected default or real")
if variant == "real" and arch != "v2.5":
    # Only the 2.5 line ships a real-data checkpoint. Falling through would
    # convert the *default* weights and report the real variant's name.
    raise SystemExit(
        f"--variant real is only available for --arch v2.5 (got {arch!r}); "
        "Prior Labs ships no real-data checkpoint for the other generations")

task = argv[0] if argv else "classification"
cache_dir = pathlib.Path(argv[1]).expanduser() if len(argv) > 1 \
    else pathlib.Path.home() / ".cache/anofox-tabfm"
which = {"classification": "classifier", "regression": "regressor"}[task]

# The cache-relative directory the safetensors lands in. `real` gets its own so
# it cannot collide with the default checkpoint of the same repo (see docstring).
task_dir = f"{task}-real" if variant == "real" else task

if arch == "v2.5":
    slug_name = "tabpfn25"
    repo_slug = "Prior-Labs__tabpfn_2_5"
    # Every revision below is pinned like its registry entry (src/tabfm_registry.cpp), so the download
    # and the cache dir match the engine's.
    revision = "6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b"
    hf_file = f"tabpfn-v2.5-{which}-v2.5_{variant}.ckpt"
    hf_url = f"https://huggingface.co/Prior-Labs/tabpfn_2_5/resolve/{revision}/{hf_file}"
elif arch == "v2.6":
    # 2.6 is its own architecture (rmsnorm, deeper per-layer parameterisation)
    # but exports through the 2.5 patch surface — see tabpfn_patched.ARCHES.
    slug_name = "tabpfn26"
    repo_slug = "Prior-Labs__tabpfn_2_6"
    revision = "24148873a9a5992b429833d07363690dcffd02a8"
    hf_file = f"tabpfn-v2.6-{which}-v2.6_default.ckpt"
    hf_url = f"https://huggingface.co/Prior-Labs/tabpfn_2_6/resolve/{revision}/{hf_file}"
elif arch == "v3":
    # tabpfn_patched.load_real_model has handled arch="v3" all along; only this
    # script's slug table did not, so `--arch=v3` fell into the v2 branch and
    # produced v2 weights while printing a v3-shaped success line.
    slug_name = "tabpfn3"
    repo_slug = "Prior-Labs__tabpfn_3"
    revision = "24a16a89d245878b846555110985634aa2e656d7"
    hf_file = f"tabpfn-v3-{which}-v3_default.ckpt"
    hf_url = f"https://huggingface.co/Prior-Labs/tabpfn_3/resolve/{revision}/{hf_file}"
else:
    slug_name = "tabpfn"
    repo_slug = {"classification": "Prior-Labs__TabPFN-v2-clf",
                 "regression": "Prior-Labs__TabPFN-v2-reg"}[task]
    revision = {"classification": "275f19d2add9c2c0de38f9ebaded3856599fa2e7",
                "regression": "02a44705ff7790b50cded404bc59adb94da456a1"}[task]
    hf_file = {"classification": "tabpfn-v2-classifier.ckpt", "regression": "tabpfn-v2-regressor.ckpt"}[task]
    hf_url = f"https://huggingface.co/{repo_slug.replace('__', '/')}/resolve/{revision}/{hf_file}"

tmap_path = (pathlib.Path(__file__).resolve().parents[2]
             / f"resources/tensor_map_{slug_name}_{task}.json")
tmap = json.load(open(tmap_path))
inits = tmap.get("initializers", tmap)  # onnx-init-name -> checkpoint key
want_keys = set(inits.values())  # checkpoint-namespace keys the graph references

tmp = pathlib.Path(tempfile.mkdtemp())
# Fetched from the pinned revision for every arch (v2 used to go through tabpfn's download_model, which
# follows the repo's main branch). The Prior Labs repos carry `extra_gated_fields`, but the resolve
# endpoint serves anonymously; if that ever changes this download 401s and the user needs an HF token
# (see docs/REAL_MODELS.md).
import urllib.request
ckpt = tmp / hf_file
print(f"downloading {hf_file} @ {revision[:8]} (transient) ...", flush=True)
urllib.request.urlretrieve(hf_url, ckpt)

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

out = cache_dir / f"{repo_slug}@{revision}" / task_dir / "model.safetensors"
out.parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(out))
print(f"wrote {len(tensors)} tensors -> {out} ({out.stat().st_size} bytes)", flush=True)
import shutil; shutil.rmtree(tmp, ignore_errors=True)
