# tools/gpu_test — exercising GPU code paths without a GPU

The dev box has no NVIDIA card, so CUDA-only behaviour cannot be tested locally.
These two scripts make it testable anyway: one rents a GPU by the minute and
runs a command on it, the other runs a committed weight-free graph through an
execution provider and reports what happened.

**Not wired into CI.** It spends money and needs a key CI does not have. Run it
by hand when a change touches a GPU path.

## `ort_ep_check.py` — run a graph on a provider

Synthesizes deterministic random initializers for the weight-free graphs in
`resources/` and injects them the way the C++ engine does, so the graph's
*structure* is exercised on the provider — which is where EP-specific bugs live
(issue #21).

```bash
# locally, on CPU or on this box's ROCm
tools/gpu_test/ort_ep_check.py resources/graph_tabicl_classification.onnx --provider cpu
tools/gpu_test/ort_ep_check.py resources/graph_tabicl_classification.onnx --provider migraphx

# two graphs must agree, ignoring outputs only one of them has
tools/gpu_test/ort_ep_check.py a.onnx b.onnx --provider cuda --compare
```

Exit codes: `0` ran (and matched, with `--compare`), `1` a run failed, `2`
outputs differed. `--opt-level` and `--no-mem-pattern` exist for isolating ORT
optimiser and allocator behaviour — both were used to eliminate hypotheses in
issue #21.

## `ort_repro.py` — the standalone upstream reproducer

Attached to [onnxruntime#32083](https://github.com/microsoft/onnxruntime/issues/32083).
Self-contained: fetches the public weight-free graph over HTTPS, synthesizes
initializers, runs it on both providers. Nothing from this repo is needed
besides the file itself, so ORT maintainers can run it directly.

```bash
pip install onnxruntime-gpu==1.23.2 onnx numpy 'nvidia-cudnn-cu12<10'
python ort_repro.py            # pre-fix graph:  CPU ok, CUDA fails at ScatterND
python ort_repro.py --pinned   # bounds pinned as outputs: both providers ok
```

Keep the URLs in it working: the upstream issue tells maintainers to `curl`
this file from `main`.

## `equivalence.py` — a device switch must not change the answer

The contract behind `docs/DYNAMIC_BACKENDS.md`. Runs the same graph on several
providers and compares each against the CPU result, which is the reference by
definition — it is the one every user gets.

```bash
# CPU self-check, synthesized weights (this is what CI can run)
tools/gpu_test/equivalence.py resources/graph_tabicl_classification.onnx --providers cpu

# the stronger statement: the real checkpoint
tools/gpu_test/equivalence.py resources/graph_tabicl_classification.onnx --providers cpu \
    --weights ~/.cache/anofox-tabfm/jingang__TabICL@main/classification/model.safetensors \
    --tensor-map resources/tensor_map_tabicl_classification.json

# on a GPU box: ONNX Runtime's own CUDA execution provider
tools/gpu_test/equivalence.py resources/graph_tabicl_classification.onnx --providers cpu,cuda
```

### Measuring what ships, not what ONNX Runtime does

A backend suffixed `-plugin` is driven through the **real backend plugin**
(`tabfm_plugin_abi.h`, `dlopen`'d from `--plugin-dir`) instead of through ORT's
execution provider. On GPU that distinction is the whole point: since
`docs/DYNAMIC_BACKENDS.md` phases 1 and 3, neither ROCm nor CUDA reaches the
accelerator through this process's ORT, so `--providers cpu,cuda` measures
ONNX Runtime while `--providers cpu,cuda-plugin` measures the shipped path.

```bash
# ROCm, against a locally built plugin
tools/gpu_test/equivalence.py resources/graph_migraphx_classification.onnx \
    --providers cpu,rocm-plugin --plugin-dir build/debug/extension/anofox_tabfm

# CUDA, against what tabfm_download_runtime('cuda') fetched
tools/gpu_test/equivalence.py resources/graph_ext_classification.onnx \
    --providers cpu,cuda-plugin --plugin-dir ~/.cache/anofox-tabfm/ep
```

Verified on real hardware: `cpu` vs `rocm-plugin` on gfx1201 with the
classification graph and synthesized weights gives `max relative 1.034e-05`
(tolerance `1e-04`) with argmax agreement `1.0000`. Expect the first run of a
new shape bucket to take tens of minutes — MIGraphX compiles per bucket, and a
synthesized-weights run deliberately cannot reuse a cached `.mxr` (see
`plugin_cache_dir`).

The plugins only speak the 5-input tabfm signature (`x, y, cat_mask,
train_size, d`), so pair them with a `graph_ext_*` / `graph_migraphx_*` graph —
the `(x, y)` graphs are ORT-provider only. The signature is detected from the
graph rather than assumed, and both sides are fed the same initializer values
(materialized to an external-data file for the plugin, which has no way to
receive them in memory).

Exit codes: `0` equivalent, `1` a backend failed, `2` answers diverged, `3` a
requested provider was not available. **Unavailable is never silent** — a GPU
comparison that quietly ran on CPU would pass while testing nothing, which is
the failure mode this whole directory exists to prevent.

## `usecase_equivalence.sql` — the same query on CPU and on a GPU

The other checks here drive a plugin from a C++ host with a handful of
hand-built rows. This one goes through the surface a user actually touches:
`SET anofox_tabfm_device` -> backend dispatch -> `ep_path` -> the plugin ->
the aggregate's chunking, against the real cached weights, comparing the
predicted labels row for row.

```bash
{ echo "SET anofox_tabfm_ep_path='$PWD/build/debug/extension/anofox_tabfm';"; \
  cat tools/gpu_test/usecase_equivalence.sql; } | ./build/debug/duckdb
```

Result on gfx1201 with the real TabFM v1 classification weights: 100 rows,
30 predicted, **0 disagreements**, and the same non-degenerate class spread on
both backends (churn 33 / stay 32 / upgrade 35).

Sized to stay inside the already-compiled T128/H16 shape bucket (<=128 rows,
<=16 features) so it needs no MIGraphX recompile. Going outside that bucket is
a fine thing to test, but budget ~27 min for the first run of each new bucket.

Needs a GPU and a real model cache, so it is a manual check rather than a CI
one — and it is the check that found two bugs no unit test could: the GPU
probes were compiled out of non-GPU flavors, so a cpu build could not see a
card at all, and `rocm` was refused on such a build even with its plugin
installed.

## `cuda_plugin_verify.sh` — the CUDA plugin on rented hardware

Builds `src/tabfm_cuda_plugin.cpp` against a real ORT-GPU distribution, loads
it through the plugin ABI, compares CPU vs CUDA on the committed fixture, and
checks the artifacts `tabfm_download_runtime('cuda')` declares against the real
wheel (byte count, entry names, SONAME).

```bash
tools/gpu_test/runpod_run.py --upload tools/gpu_test/cuda_plugin_verify.sh \
    tools/gpu_test/cuda_plugin_verify_host.cpp src/tabfm_cuda_plugin.cpp \
    src/include/tabfm_plugin_abi.h test/fixtures/graph_fixture.onnx \
    test/fixtures/model.safetensors \
    --command 'bash /workspace/cuda_plugin_verify.sh'
```

## `runpod_run.py` — rent a GPU, run, destroy

```bash
# the issue #21 regression: pre-pin graphs must fail, pinned graphs must pass
tools/gpu_test/runpod_run.py --check-scatternd

# anything else
tools/gpu_test/runpod_run.py --with-ort --upload some.onnx \
    --command 'python /workspace/ort_ep_check.py /workspace/some.onnx --provider cuda'

tools/gpu_test/runpod_run.py --list                 # pods still running
tools/gpu_test/runpod_run.py --terminate <pod-id>   # clean up after a SIGKILL
```

Picks the cheapest rentable GPU by default (an EP-level graph bug reproduces on
any CUDA device, and these graphs are small), falling through the price-sorted
list and both clouds when one is sold out.

`RUNPOD_API_KEY` comes from the environment or, failing that, is read out of
`~/.bashrc` — which exports it but returns early for non-interactive shells, so
sourcing it yields nothing. The value is never printed.

### Cost

The pod is terminated in a `finally` block and on SIGINT/SIGTERM. A SIGKILL
leaks it — `--list` and `--terminate` exist for that. Check `--list` after an
interrupted run.

### Two traps this harness exists to avoid

Both make a GPU run silently measure **CPU** while looking like it passed:

- `nvidia-cudnn-cu12` unpinned resolves to cuDNN 10, which ORT 1.2x cannot load,
  so the CUDA provider library fails to `dlopen`. Pinned to `<10` here.
- `get_available_providers()` reports what the ORT *build* supports, not what
  can be instantiated — it lists `CUDAExecutionProvider` even when loading it
  fails. `--with-ort` therefore builds a probe session and refuses to continue
  unless the session really came up on CUDA.

Three runs during issue #21 measured CPU before this was caught.
