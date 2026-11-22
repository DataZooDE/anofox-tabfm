#!/usr/bin/env python3
"""The plugin release must publish exactly what tabfm_download_runtime fetches (src/include/tabfm_plugin_artifacts.hpp:
PluginFileNameFor + PluginSha256AssetName), every file built by a job the publish step waits for, and nothing of ONNX Runtime.
Usage: scripts/check_plugin_release_assets.py [.github/workflows/gpu_plugins.yml]"""
import re, sys

path = sys.argv[1] if len(sys.argv) > 1 else ".github/workflows/gpu_plugins.yml"
text = open(path).read()

# The names the extension asks for (one source of truth: mirrored here and pinned by test_tabfm_plugin_backend.cpp).
expected = {
    "libanofox_tabfm_ggml-cuda_plugin.so", "ggml-cuda_plugin.sha256",
    "libanofox_tabfm_ggml-rocm_plugin.so", "ggml-rocm_plugin.sha256",
    "libanofox_tabfm_mlx_plugin.dylib", "mlx_plugin.sha256",
}
errors = []

upload = re.search(r"^[ \t]*gh release upload \"\$\{GITHUB_REF_NAME\}\"[^\n]*\n((?:[ \t]+[^\n]*\\\n)*[ \t]+[^\n]*)", text, re.M)
if not upload:
    errors.append("no `gh release upload` step found")
    published = set()
else:
    published = {t for t in re.split(r"[\s\\]+", upload.group(1)) if t and not t.startswith("-") and not t.startswith("$") and "REPOSITORY" not in t}
for name in sorted(expected - published):
    errors.append(f"publish does not upload {name}")
for name in sorted(published - expected):
    errors.append(f"publish uploads {name}, which no tabfm_download_runtime call fetches")

jobs = dict(re.findall(r"\n  ([a-z0-9-]+):\n    name: ([^\n]*)", text))
needs = re.search(r"publish:.*?needs: \[([^\]]*)\]", text, re.S)
needed = {n.strip() for n in needs.group(1).split(",")} if needs else set()
blocks = re.split(r"\n  (?=[a-z0-9-]+:\n    name:)", text)
for name in sorted(expected):
    producers = [b.split(":", 1)[0] for b in blocks if "upload-artifact" in b and re.search(rf"^\s+{re.escape(name)}\s*$", b, re.M)]
    if not producers:
        errors.append(f"no job uploads {name} as an artifact")
    elif not set(producers) & needed:
        errors.append(f"{name} is built by {producers}, which publish does not wait for (needs: {sorted(needed)})")

for ort in ("ORT_GPU", "onnxruntime", "libanofoxort_gpu", "migraphx_plugin", "tabfm_cuda_plugin.cpp"):
    if ort in text:
        errors.append(f"the workflow still references {ort!r} (the ORT-CUDA and MIGraphX plugins are gone)")

# The ROCm architectures the plugin is built for must be exactly the ones the extension reports usable (GgmlRocmPluginArchs in
# src/tabfm_devices.cpp): a target built but not listed is a card refused for nothing, one listed but not built is a card chosen and then failing.
targets = re.search(r"-DAMDGPU_TARGETS=([^\s\"']+)", text)
built = set(targets.group(1).split(";")) if targets else set()
src = open("src/tabfm_devices.cpp").read()
listed_m = re.search(r"GgmlRocmPluginArchs\(\) \{\s*static const vector<string> archs = \{([^}]*)\}", src)
listed = set(re.findall(r'"(gfx[0-9a-f]+)"', listed_m.group(1))) if listed_m else set()
if not built:
    errors.append("the ROCm job sets no -DAMDGPU_TARGETS")
if not listed:
    errors.append("GgmlRocmPluginArchs() not found in src/tabfm_devices.cpp")
if built and listed and built != listed:
    errors.append(f"AMDGPU_TARGETS {sorted(built)} != GgmlRocmPluginArchs {sorted(listed)}")

if errors:
    print("FAIL: " + path)
    for e in errors:
        print("  - " + e)
    sys.exit(1)
print(f"OK: {path} publishes exactly {sorted(expected)}")
