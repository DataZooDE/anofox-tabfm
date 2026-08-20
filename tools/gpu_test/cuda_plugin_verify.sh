#!/usr/bin/env bash
# Pod verification for the CUDA backend plugin (docs/DYNAMIC_BACKENDS.md phase 3).
#
#   1. build src/tabfm_cuda_plugin.cpp against a real ORT-GPU distribution
#   2. load it through the real plugin ABI and run the committed fixture on
#      CUDA, comparing against a CPU ORT run of the same graph
#   3. verify the artifacts tabfm_download_runtime('cuda') actually fetches:
#      the three wheel entries exist and the core carries the SONAME the
#      plugin links against
set -uo pipefail
cd /workspace
FAILED=0

echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

ORT_VER=1.28.0
ARCHIVE=onnxruntime-linux-x64-gpu_cuda12-${ORT_VER}
echo
echo "=== 1. fetch the ORT-GPU archive ==="
curl -sSL -o ort_gpu.tgz \
  "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${ARCHIVE}.tgz" || exit 1
tar xzf ort_gpu.tgz
ORT_DIR=/workspace/${ARCHIVE}
ls "${ORT_DIR}/lib/" | sed 's/^/    /'

echo
echo "=== 2. synthesize the fixture's external-data file ==="
# The committed fixture graph references graph_fixture.onnx.data, which is not
# committed (the engine normally injects initializers by name instead). Rebuild
# it from the committed random-init model.safetensors so the plugin's
# external-data path can be exercised without any real weights present
# (CLAUDE.md's license wall).
mkdir -p /workspace/fixture
cp /workspace/graph_fixture.onnx /workspace/model.safetensors /workspace/fixture/
python3 -m pip install -q onnx 2>&1 | tail -1
python3 - <<'PY' || exit 1
import json, struct, onnx

with open("/workspace/fixture/model.safetensors","rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    blob = f.read()

m = onnx.load("/workspace/fixture/graph_fixture.onnx", load_external_data=False)
buf = bytearray()
placed = missing = 0
for init in m.graph.initializer:
    if init.data_location != onnx.TensorProto.EXTERNAL:
        continue
    meta = {e.key: e.value for e in init.external_data}
    off, length = int(meta["offset"]), int(meta["length"])
    key = init.name[2:] if init.name.startswith("m.") else init.name  # S06 "m." convention
    if key not in header:
        missing += 1
        continue
    b0, b1 = header[key]["data_offsets"]
    src = blob[b0:b1]
    if len(src) != length:
        print(f"  size mismatch for {init.name}: graph wants {length}, safetensors has {len(src)}")
        missing += 1
        continue
    if len(buf) < off + length:
        buf.extend(b"\0" * (off + length - len(buf)))
    buf[off:off+length] = src
    placed += 1

with open("/workspace/fixture/graph_fixture.onnx.data","wb") as f:
    f.write(buf)
print(f"  placed {placed} tensors, {missing} unmatched, {len(buf)} bytes")
if missing or not placed:
    raise SystemExit(1)
PY

echo
echo "=== 3. build the plugin against the ORT-GPU distribution ==="
g++ -O2 -std=c++17 -shared -fPIC -o libanofox_tabfm_cuda_plugin.so \
    tabfm_cuda_plugin.cpp \
    -I/workspace -I"${ORT_DIR}/include" \
    -L"${ORT_DIR}/lib" -lonnxruntime \
    -Wl,-rpath,'$ORIGIN' || { echo "PLUGIN BUILD FAILED"; exit 1; }
echo "  built. DT_NEEDED / RPATH:"
readelf -d libanofox_tabfm_cuda_plugin.so | grep -E "NEEDED.*onnxruntime|RUNPATH|RPATH" | sed 's/^/    /'
# $ORIGIN must resolve the core: stage it next to the plugin, as
# tabfm_download_runtime('cuda') does.
cp -P "${ORT_DIR}/lib/"libonnxruntime.so* /workspace/
echo "  ldd:"
ldd libanofox_tabfm_cuda_plugin.so | grep -i onnxruntime | sed 's/^/    /'

echo
echo "=== 4. load through the plugin ABI and compare CPU vs CUDA, per precision mode ==="
g++ -O2 -std=c++17 -o verify_host cuda_plugin_verify_host.cpp \
    -I/workspace -I"${ORT_DIR}/include" \
    -L"${ORT_DIR}/lib" -lonnxruntime -ldl \
    -Wl,-rpath,"${ORT_DIR}/lib" || { echo "HOST BUILD FAILED"; exit 1; }
# Track A: fp32 must be strict (use_tf32=0) and agree with CPU; tf32 opts the
# tensor-core rounding back in (agreement may loosen slightly, both printed for
# the plan's cost table); bf16 must be REJECTED at create with the CUDA
# message. A silent fp32 run for bf16 is the failure this exists to catch.
for MODE in fp32 tf32 bf16; do
  echo "  --- precision=$MODE ---"
  ./verify_host "$MODE"
  RC=$?
  [ $RC -ne 0 ] && FAILED=1
  echo "  verify_host($MODE) exit=${RC}"
done

echo
echo "=== 5. verify what tabfm_download_runtime('cuda') fetches ==="
WHEEL_URL="https://aiinfra.pkgs.visualstudio.com/2692857e-05ef-43b4-ba9c-ccf1c22c437c/_packaging/9387c3aa-d9ad-4513-968c-383f6f7f53b8/pypi/download/onnxruntime-gpu/1.28/onnxruntime_gpu-1.28.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
curl -sSL -o rt.whl "$WHEEL_URL" || { echo "WHEEL DOWNLOAD FAILED"; FAILED=1; }
if [ -f rt.whl ]; then
  echo "  wheel bytes: $(stat -c %s rt.whl) (src/tabfm_weights.cpp declares 432340836)"
  python3 - <<'PY'
import zipfile, sys
want = ["onnxruntime/capi/libonnxruntime.so.1.28.0",
        "onnxruntime/capi/libonnxruntime_providers_cuda.so",
        "onnxruntime/capi/libonnxruntime_providers_shared.so"]
z = zipfile.ZipFile("/workspace/rt.whl")
names = set(z.namelist())
ok = True
for w in want:
    present = w in names
    print(f"    {'OK  ' if present else 'MISS'} {w}")
    ok = ok and present
if ok:
    # the core must land under the SONAME the plugin's DT_NEEDED asks for
    with open("/workspace/extracted_core.so","wb") as f:
        f.write(z.read(want[0]))
sys.exit(0 if ok else 1)
PY
  [ $? -ne 0 ] && FAILED=1
  if [ -f extracted_core.so ]; then
    SONAME=$(readelf -d extracted_core.so | grep -i soname | sed 's/.*\[\(.*\)\].*/\1/')
    echo "  extracted core SONAME: ${SONAME} (tabfm_weights.cpp writes it as libonnxruntime.so.1)"
    [ "$SONAME" = "libonnxruntime.so.1" ] || { echo "  SONAME MISMATCH"; FAILED=1; }
  fi
fi

echo
if [ $FAILED -eq 0 ]; then
  echo "POD_VERIFY_RESULT: PASSED"
else
  echo "POD_VERIFY_RESULT: FAILED"
fi
echo "POD_VERIFY_DONE_MARKER"
exit $FAILED
