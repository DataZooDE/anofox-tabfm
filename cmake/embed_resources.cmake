# Embed the weight-free artifacts in resources/ into the extension binary.
#
# Generates ${CMAKE_CURRENT_BINARY_DIR}/tabfm_bundled_resources.cpp implementing
# GetBundledResource() (see src/include/tabfm_bundled_resources.hpp) from the
# committed .onnx graphs + tensor maps, so the built-in model flow needs no
# companion files on disk. The bytes are Google-weight-free (license wall:
# S01/S06 — graphs are weight-free, tensor maps are name mappings only).
#
# Each file is registered under its filename; graphs additionally under their
# bare stem ("graph_classification") — the id the built-in manifest uses. A
# filesystem path never matches, so path-valued manifests fall through to disk.
#
# Output: sets TABFM_BUNDLED_RESOURCES_SRC to the generated source path.

set(_tabfm_res_dir "${CMAKE_CURRENT_SOURCE_DIR}/resources")
set(_tabfm_gen "${CMAKE_CURRENT_BINARY_DIR}/tabfm_bundled_resources.cpp")
set(_tabfm_inputs
    "graph_classification.onnx"
    "graph_regression.onnx"
    "graph_ext_classification.onnx"
    "graph_ext_regression.onnx"
    "graph_migraphx_classification.onnx"
    "graph_migraphx_regression.onnx"
    "tensor_map_classification.json"
    "tensor_map_regression.json"
    # Built-in catalog models (Mitra, TabPFN v2, TabICL v2) — bundled so they are
    # usable by name with no manifest file (pure-SQL surface).
    "graph_mitra_classification.onnx"
    "graph_mitra_regression.onnx"
    # Mitra GPU variants (tools/make_external_graph.py / make_migraphx_graph.py
    # against the HF weights; gated by ExpectedWeightsHeaderShaFor at load).
    "graph_ext_mitra_classification.onnx"
    "graph_ext_mitra_regression.onnx"
    "graph_migraphx_mitra_classification.onnx"
    "graph_migraphx_mitra_regression.onnx"
    # TabDPT MIGraphX graphs. The only single_eval_pos model with them: its
    # train/test split was converted from a positional slice to a train_size
    # mask (tools/export_tabdpt, docs/ROCM_TABDPT_SPIKE.md), which makes every
    # shape a function of the (T, H) bucket and so compilable by MIGraphX.
    # Larger than mitra's (5.9 MB vs 0.6 MB): the masked form has more nodes,
    # and the exporter's optimize=True would halve it but renames initializers
    # out of the tensor map, which make_migraphx_graph.py refuses outright.
    "graph_migraphx_tabdpt_classification.onnx"
    "graph_migraphx_tabdpt_regression.onnx"
    # TabPFN v2 / v2.5 / v2.6 MIGraphX graphs, both tasks (tools/export_tabpfn
    # --contract mask; docs/ROCM_TABPFN_PLAN.md). Unlike tabdpt's they take a
    # FIFTH input, `n_rows`, the real row count before bucket padding: TabPFN's
    # constant-feature detection and feature-group normalisation compare every
    # row against row 0, so a padded row changes the answer. The MIGraphX plugin
    # binds it only when the graph declares it, so the older graphs are untouched
    # -- and an OLDER PLUGIN refuses these ("Parameter not found: n_rows"), which
    # the engine turns into an instruction to update it.
    # tabpfn-v2-5-real reuses the 2.5 graphs (see BundledGpuGraphId).
    # TabPFN-3 too, with two differences: it carries NO inline initializer at all
    # (RoPE, so there is no positional table to keep inline), and its cold
    # compile is the longest of the family, ~16 min per shape bucket on gfx1201.
    "graph_migraphx_tabpfn_classification.onnx"
    "graph_migraphx_tabpfn_regression.onnx"
    "graph_migraphx_tabpfn25_classification.onnx"
    "graph_migraphx_tabpfn25_regression.onnx"
    "graph_migraphx_tabpfn26_classification.onnx"
    "graph_migraphx_tabpfn26_regression.onnx"
    "graph_migraphx_tabpfn3_classification.onnx"
    "graph_migraphx_tabpfn3_regression.onnx"
    # Catalog ext graphs (CUDA + CPU low-memory). Models with no migraphx
    # variant are refused on ROCm by name — see ExpectedWeightsHeaderShaFor's
    # comment, docs/ROCM_SINGLE_EVAL_POS.md and tabfm_backends().
    "graph_ext_tabpfn_classification.onnx"
    "graph_ext_tabpfn_regression.onnx"
    "graph_ext_tabpfn25_classification.onnx"
    "graph_ext_tabpfn25_regression.onnx"
    "graph_ext_tabpfn26_classification.onnx"
    "graph_ext_tabpfn26_regression.onnx"
    "graph_ext_tabpfn3_classification.onnx"
    "graph_ext_tabpfn3_regression.onnx"
    "graph_ext_tabicl_classification.onnx"
    "graph_ext_tabicl_regression.onnx"
    "graph_ext_orion_bix_classification.onnx"
    "graph_ext_orion_msp_classification.onnx"
    "graph_ext_tabdpt_classification.onnx"
    "graph_ext_tabdpt_regression.onnx"
    "graph_tabpfn_classification.onnx"
    "graph_tabpfn_regression.onnx"
    "graph_tabicl_classification.onnx"
    "graph_tabicl_regression.onnx"
    "tensor_map_mitra_classification.json"
    "tensor_map_mitra_regression.json"
    "tensor_map_tabpfn_classification.json"
    "tensor_map_tabpfn_regression.json"
    "tensor_map_tabicl_classification.json"
    "tensor_map_tabicl_regression.json"
    # Causilo (Nums AI, non-commercial weights) -- (x, y)-only like TabICL, so one plain graph and
    # one tensor map per task and nothing else: the CPU path on every platform, no ext graph yet
    # (CUDA), no migraphx graph (ROCm; its split is positional, see docs/ROCM_SINGLE_EVAL_POS.md).
    "graph_causilo_classification.onnx"
    "graph_causilo_regression.onnx"
    # The external-data variants: on CUDA, and on the CPU default path, ORT reads weights off the
    # cached safetensors by offset instead of copying them in (src/tabfm_engine.cpp
    # TryExternalDataSession). Valid only for the revision the registry pins; the header hash in
    # ExpectedWeightsHeaderShaFor decides at load time whether to use them.
    "graph_ext_causilo_classification.onnx"
    "graph_ext_causilo_regression.onnx"
    "tensor_map_causilo_classification.json"
    "tensor_map_causilo_regression.json"
    # LimiX-2M (StableAI; commercial under the Stable AI Technology Co., Ltd. License v1.0) -- (x, y)-only,
    # one plain graph and one tensor map per task, plus the CUDA/CPU external-data variants. NO migraphx
    # graph: its split is positional, so ROCm refuses it by name (docs/ROCM_SINGLE_EVAL_POS.md). The stem is
    # "limix2m" so LimiX-2 (400M) can never be mistaken for it.
    "graph_limix2m_classification.onnx"
    "graph_limix2m_regression.onnx"
    "graph_ext_limix2m_classification.onnx"
    "graph_ext_limix2m_regression.onnx"
    "tensor_map_limix2m_classification.json"
    "tensor_map_limix2m_regression.json"
    # TabPFN-2.5 (Prior Labs, non-commercial) — per-task graphs AND maps.
    "graph_tabpfn25_classification.onnx"
    "graph_tabpfn25_regression.onnx"
    # TabPFN-2.6 (Prior Labs, non-commercial) — its own graphs AND maps: the 2.6
    # architecture maps 322/324 initializers where 2.5 maps 250/192.
    "graph_tabpfn26_classification.onnx"
    "graph_tabpfn26_regression.onnx"
    "tensor_map_tabpfn26_classification.json"
    "tensor_map_tabpfn26_regression.json"
    "tensor_map_tabpfn25_classification.json"
    "tensor_map_tabpfn25_regression.json"
    # Orion-BiX (MIT) — classification only, no regression graph upstream.
    "graph_orion_bix_classification.onnx"
    "tensor_map_orion_bix_classification.json"
    # Orion-MSP (MIT) — classification only, like its Orion-BiX sibling.
    "graph_orion_msp_classification.onnx"
    "tensor_map_orion_msp_classification.json"
    # TabDPT (Layer 6 AI, Apache-2.0) — both tasks off one checkpoint.
    "graph_tabdpt_classification.onnx"
    "graph_tabdpt_regression.onnx"
    "tensor_map_tabdpt_classification.json"
    "tensor_map_tabdpt_regression.json"
    # TabPFN-3 (Prior Labs, non-commercial) — a different architecture again:
    # distribution-embedding + feature-aggregation stacks with RoPE positions.
    "graph_tabpfn3_classification.onnx"
    "graph_tabpfn3_regression.onnx"
    "tensor_map_tabpfn3_classification.json"
    "tensor_map_tabpfn3_regression.json")

# Re-run configure when a resource changes so the generated source stays fresh.
foreach(_f ${_tabfm_inputs})
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_tabfm_res_dir}/${_f}")
endforeach()

# Regenerate only when missing or stale (the hex embedding is a few seconds).
#
# "Stale" includes THIS FILE changing, not only an input being newer. Adding an entry to the list
# above whose resource is OLDER than the generated source (the usual order: export the graph, then
# register it) left the stale source in place, so the new resource silently was not embedded and
# GetBundledResource() returned null. For an ext graph that means the engine quietly takes the
# injection path instead: no error, same answers, wrong path. A fresh CI build never sees it.
set(_tabfm_stale FALSE)
if(NOT EXISTS "${_tabfm_gen}")
    set(_tabfm_stale TRUE)
elseif("${CMAKE_CURRENT_LIST_FILE}" IS_NEWER_THAN "${_tabfm_gen}")
    set(_tabfm_stale TRUE)
else()
    foreach(_f ${_tabfm_inputs})
        if("${_tabfm_res_dir}/${_f}" IS_NEWER_THAN "${_tabfm_gen}")
            set(_tabfm_stale TRUE)
        endif()
    endforeach()
endif()

if(_tabfm_stale)
    message(STATUS "anofox_tabfm: embedding resources -> ${_tabfm_gen}")
    set(_body "// Generated by cmake/embed_resources.cmake — do not edit.\n")
    string(APPEND _body "#include \"tabfm_bundled_resources.hpp\"\n\n")
    string(APPEND _body "namespace duckdb {\nnamespace anofox {\nnamespace {\n\n")
    set(_idx 0)
    set(_table "")
    foreach(_f ${_tabfm_inputs})
        set(_path "${_tabfm_res_dir}/${_f}")
        if(NOT EXISTS "${_path}")
            message(FATAL_ERROR "anofox_tabfm: bundled resource missing: ${_path}")
        endif()
        file(READ "${_path}" _hex HEX)
        string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _arr "${_hex}")
        string(APPEND _body "static const unsigned char kRes${_idx}[] = {${_arr}};\n\n")
        string(APPEND _table "    {\"${_f}\", kRes${_idx}, sizeof(kRes${_idx})},\n")
        if(_f MATCHES "\\.onnx$")
            string(REGEX REPLACE "\\.onnx$" "" _stem "${_f}")
            string(APPEND _table "    {\"${_stem}\", kRes${_idx}, sizeof(kRes${_idx})},\n")
        endif()
        math(EXPR _idx "${_idx}+1")
    endforeach()
    string(APPEND _body "struct BundledEntry {\n")
    string(APPEND _body "    const char *id;\n    const unsigned char *data;\n    unsigned long size;\n};\n\n")
    string(APPEND _body "const BundledEntry kBundled[] = {\n${_table}};\n\n")
    string(APPEND _body "} // namespace\n\n")
    string(APPEND _body "BundledResource GetBundledResource(const string &id) {\n")
    string(APPEND _body "\tfor (const auto &e : kBundled) {\n")
    string(APPEND _body "\t\tif (id == e.id) {\n")
    string(APPEND _body "\t\t\treturn BundledResource{reinterpret_cast<const char *>(e.data), static_cast<idx_t>(e.size)};\n")
    string(APPEND _body "\t\t}\n\t}\n")
    string(APPEND _body "\treturn BundledResource{nullptr, 0};\n}\n\n")
    string(APPEND _body "} // namespace anofox\n} // namespace duckdb\n")
    file(WRITE "${_tabfm_gen}" "${_body}")
endif()

set(TABFM_BUNDLED_RESOURCES_SRC "${_tabfm_gen}")
