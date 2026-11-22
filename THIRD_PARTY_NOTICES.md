# Third-party notices

anofox-tabfm bundles weight-free model graphs (ONNX files in `resources/`, compiled into the extension
binary). Model weights are never bundled; `tabfm_download` fetches them from their publishers. Where a
model's licence asks whoever distributes part of the model to keep a notice or an agreement with it, that
text is listed here, and `tabfm_list_models()` shows it in the `attribution` column.

## NVIDIA Kumo-Tabular (`kumo-tabular-small`, `-medium`, `-large`)

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

The weights are licensed under the OpenMDW License Agreement, version 1.1 (OpenMDW-1.1). The agreement
defines "Model Materials" to include a model's architecture, so the graphs bundled here, which encode
Kumo-Tabular's architecture without its parameters, may be a portion of the Model Materials. We distribute
them under that reading: the agreement is retained verbatim in
[`resources/license_openmdw_1_1_nvidia.txt`](resources/license_openmdw_1_1_nvidia.txt), copied from the
`LICENSE` file of [nvidia/Kumo-Tabular](https://huggingface.co/nvidia/Kumo-Tabular) at revision
`b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f` (SHA-256
`44b43ed7b8e0f987dc5e90fb6cb4b901eccf0b14dc9f17a9033951603851d37d`), and the binary carries the same text.
The Kumo-Tabular code is Apache-2.0 (NVIDIA/structured-data-models).

This file records how we comply; it is not legal advice.

## ggml (`vendor/ggml`)

Copyright (c) 2023-2026 The ggml authors. MIT License; the full text is in [`vendor/ggml/LICENSE`](vendor/ggml/LICENSE).
ggml is compiled into the extension as the CPU inference engine (static, CPU backend only). Its source is a pinned
git submodule; no model weights or graphs come from it.

## GPU backend plugins (release assets, not part of the extension binary)

No build of the extension or of any plugin links ONNX Runtime or MIGraphX any more.

- **ggml CUDA plugin** (`libanofox_tabfm_ggml-cuda_plugin.so`): ggml (above) with its CUDA backend, plus the
  NVIDIA CUDA runtime (`cudart`) and cuBLAS / cuBLASLt from CUDA Toolkit 12.8, **linked statically**. These
  are components NVIDIA lets applications redistribute under the CUDA Toolkit End User License Agreement
  (Attachment A). The plugin loads the NVIDIA driver (`libcuda.so.1`) from the user's system and ships no
  driver code.
- **ggml ROCm plugin** (`libanofox_tabfm_ggml-rocm_plugin.so`): ggml with its HIP backend. It links HIP,
  hipBLAS and rocBLAS dynamically from the user's ROCm installation and bundles none of them.
- **MLX plugin** (`libanofox_tabfm_mlx_plugin.dylib`): links Apple's `libmlx` / `libmlxc` (MIT) dynamically
  from the user's installation (`brew install mlx mlx-c`) and bundles neither.
