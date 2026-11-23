# tools/gpu_test — exercising GPU code paths

CUDA, ROCm and MLX are served by plugins the extension `dlopen`s (`libanofox_tabfm_ggml-cuda_plugin.so`,
`libanofox_tabfm_ggml-rocm_plugin.so`, `libanofox_tabfm_mlx_plugin.dylib`). The scripts here run the SQL
surface against a built extension and plugin on a machine with the GPU, and rent a CUDA GPU by the minute when
there is none at hand.

**Not wired into CI.** They need a GPU (and `runpod_run.py` spends money and needs a key CI does not have).
Run them by hand when a change touches a GPU path. CI covers the plugin build, its dependency check and
`plugin_load_check.c`; the device-aware SQL tests (`test/sql/tabfm_gpu_use_cases.test`,
`tabfm_gpu_registered_model.test`) run wherever `TABFM_GPU_DEVICE` / `TABFM_GPU_EP_PATH` are set.

**Read `*_served_by` before believing any number.** "cpu and gpu agree" is also what a silent CPU fallback
prints; every script here reports which backend served each side, and anything other than `cuda:N`, `rocm:N`
or `mlx:0` on the GPU side means the comparison tested nothing.

## `gpu_catalog.sh` — the whole model catalog on a GPU, against the CPU

The real-weights quality loop of `docs/GGML_PLAN.md`: every (model, task) the catalog lists, each in a FRESH
DuckDB process (a crash costs its own row), through `scenarios/cuda_catalog*.sql` with the device rewritten to
`$GPU`. Reports `served_by`, label differences, the largest score and relative regression differences, both
qualities, a repeat check and the timings.

```bash
GPU=rocm EP=build/ggml-rocm tools/gpu_test/gpu_catalog.sh
GPU=cuda EP=/workspace/ep MODELS="mitra:cr tabicl-v2:c" tools/gpu_test/gpu_catalog.sh
GPU=mlx  EP=build/release/extension/anofox_tabfm ACCEPT_LICENSES=1 tools/gpu_test/gpu_catalog.sh
```

Weights come from `CACHE` (default `~/.cache/anofox-tabfm`); missing ones are fetched with `tabfm_download`
(`ACCEPT_LICENSES=1` accepts the HuggingFace licenses for that run). Results land in `$OUT/report.csv`.

## `usecase_equivalence.sql` — the same query on CPU and on a GPU

One classification query through the surface a user touches (`SET anofox_tabfm_device` -> the plugin dispatch
-> `ep_path` -> the plugin -> the aggregate's chunking), against the real TabFM v1 weights, comparing the
predicted labels row for row and printing both class distributions (so a degenerate one-class GPU result
cannot hide behind a high agreement).

```bash
{ echo "SET anofox_tabfm_ep_path='$PWD/build/ggml-rocm';"; \
  cat tools/gpu_test/usecase_equivalence.sql; } | ./build/release/duckdb
```

It found the bugs no unit test could reach, because unit tests build device lists by hand and never touch
discovery or the SQL path: a device switch mid-session kept serving the previous device silently (70a6800, the
reason `CanReuseSession` checks the device), and nothing reported which backend served a query
(`tabfm_models().device` exists because of it, ab5c6e3).

## `scenarios/`

SQL scenarios for specific checks (fixture model on ROCm, `auto` per model, wide inputs, non-finite targets,
MLX stress, a user workflow); see `scenarios/README.md`.

## `runpod_run.py` — rent a GPU, run, destroy

Talks to RunPod's **REST API v2** (`https://api.runpod.io/v2`). The request and response shapes (pod body,
`ssh.direct`, `/network-volumes`) are pinned by `tests/test_runpod_run.py`, offline and free:
`python3 -m unittest discover -s tools/gpu_test/tests -v`.

```bash
tools/gpu_test/runpod_run.py --secure --gpu "NVIDIA GeForce RTX 4090" \
    --upload build/ggml-cuda/libanofox_tabfm_ggml-cuda_plugin.so tools/gpu_test/gpu_catalog.sh \
    --command 'nvidia-smi && ls -l /workspace'

tools/gpu_test/runpod_run.py --list                 # pods still running
tools/gpu_test/runpod_run.py --terminate <pod-id>   # clean up after a SIGKILL
```

Picks the cheapest rentable GPU by default, falling through the price-sorted list and both clouds when one is
sold out. Prefer `--secure`: community hosts have been seen where `nvidia-smi` works but CUDA cannot initialise
the device (user-namespace uid mapping).

For repeated CUDA iterations, attach a persistent network volume so uploads, toolchains and build trees survive
across pods:

```bash
tools/gpu_test/runpod_run.py --create-volume 40 --volume-dc EUR-IS-1  # RECURRING cost until deleted
tools/gpu_test/runpod_run.py --volume-id <id> --gpu "NVIDIA GeForce RTX 4090" --command ...
tools/gpu_test/runpod_run.py --delete-volume <id>                     # ends the cost
```

Volumes exist only in specific datacenters (a wrong `--volume-dc` errors with the valid list) and pin the pod to
their datacenter, where GPU stock may be scarce: pass an explicit `--gpu` for a type that datacenter stocks.

`RUNPOD_API_KEY` comes from the environment or, failing that, is read out of `~/.bashrc` (which exports it but
returns early for non-interactive shells, so sourcing it yields nothing). The value is never printed.

### Cost

The pod is terminated in a `finally` block and on SIGINT/SIGTERM. A SIGKILL leaks it: `--list` and
`--terminate` exist for that. Check `--list` after an interrupted run.
