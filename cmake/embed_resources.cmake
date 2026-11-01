# Embed the weight-free artifacts in resources/ into the extension binary.
#
# Generates ${CMAKE_CURRENT_BINARY_DIR}/tabfm_bundled_resources.cpp implementing
# GetBundledResource() (see src/include/tabfm_bundled_resources.hpp) plus ONE source file per resource
# (${CMAKE_CURRENT_BINARY_DIR}/tabfm_res/tabfm_res_<n>.cpp) holding that resource's bytes, from the
# committed .onnx graphs + tensor maps, so the built-in model flow needs no
# companion files on disk. The bytes are Google-weight-free (license wall:
# S01/S06 — graphs are weight-free, tensor maps are name mappings only).
#
# Each file is registered under its filename; graphs additionally under their
# bare stem ("graph_classification") — the id the built-in manifest uses. A
# filesystem path never matches, so path-valued manifests fall through to disk.
#
# WHY one file per resource: the bytes are emitted as `0x..,` initializer lists, and GCC 14 spends about
# 120 bytes of memory per element. All resources in ONE translation unit (68 million elements) needed 8.0 GB
# to compile under the CI image's compiler (measured in quay.io/pypa/manylinux_2_28 with gcc-toolset-14.2;
# GCC 16 needs 1.1 GB for the same file), on a runner that is also compiling DuckDB's unity-build files in
# parallel. The build was killed (exit 137) at a different step each time, and adding 3% more data (the
# LimiX-2M graphs) was enough to make it fail repeatedly. With one array per translation unit the largest
# compile is the largest resource (6 MB), well under 1 GB, and the same holds however many models are added.
# Sizes are written as literals into the index file, so its table is constant-initialised across files.
#
# Output: sets TABFM_BUNDLED_RESOURCES_SRC to a LIST of generated sources (the index and every resource).

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
    # TabDPT v1.3 (`tabdpt-v1-3`): the same masked conversion, re-exported for v1.3's per-task
    # y-encoders (tools/export_tabdpt --config real13 --contract mask).
    "graph_migraphx_tabdpt13_classification.onnx"
    "graph_migraphx_tabdpt13_regression.onnx"
    # Mitra-v2 (`mitra-v2`): the 1000-bin regressor only -- the v2 classifier reuses mitra's graphs
    # (byte-identical safetensors header). bin_edges/bin_centres stay inline (code-generated).
    "graph_mitra2_regression.onnx"
    "graph_ext_mitra2_regression.onnx"
    "graph_migraphx_mitra2_regression.onnx"
    "tensor_map_mitra2_regression.json"
    "graph_tabdpt13_classification.onnx"
    "graph_tabdpt13_regression.onnx"
    "graph_ext_tabdpt13_classification.onnx"
    "graph_ext_tabdpt13_regression.onnx"
    "tensor_map_tabdpt13_classification.json"
    "tensor_map_tabdpt13_regression.json"
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
    # Google TabFM 1.1.0 (tabfm-v1-1; non-commercial like v1). New graphs only: the dims are baked in
    # (embed_dim 128), but the tensor maps are byte-identical to v1's, so it reuses those.
    "graph_tabfm_v1_1_classification.onnx"
    "graph_tabfm_v1_1_regression.onnx"
    # Its GPU variants (+ the CPU low-memory path): the checkpoint is BF16, so each initializer is
    # retyped to BF16 and Cast to FLOAT in-graph; valid only for the pinned header (see
    # ExpectedWeightsHeaderShaFor). The MIGraphX graph serves ROCm with the same shape buckets as v1.
    "graph_ext_tabfm_v1_1_classification.onnx"
    "graph_ext_tabfm_v1_1_regression.onnx"
    "graph_migraphx_tabfm_v1_1_classification.onnx"
    "graph_migraphx_tabfm_v1_1_regression.onnx"
    # Nori-30M (Synthefy, Apache-2.0) -- regression ONLY, (x, y)-only on the CPU like its LimiX ancestor: one
    # plain graph and one tensor map. The engine reads nori.pt natively ({model_config, ema_state_dict}). The
    # migraphx graph is the MASKED contract (tools/export_nori --contract mask) and reads the safetensors that
    # tabfm_download writes beside the checkpoint (header sha in ExpectedWeightsHeaderShaFor).
    "graph_nori30m_regression.onnx"
    "graph_migraphx_nori30m_regression.onnx"
    "tensor_map_nori30m_regression.json"
    # EXAONE Tabular v1 (LG AI Research; non-commercial EXAONE NC 1.2 weights) -- (x, y)-only on the CPU, one
    # plain graph and one tensor map per task. The released files are safetensors keyed by the module
    # namespace, so there is no converter. The migraphx graphs are the MASKED contract (x, y_full, train_size,
    # d, n_rows; tools/export_exaone --contract mask), so ROCm can bucket them; they bake offsets into LG's
    # released files, checked by the header sha in ExpectedWeightsHeaderShaFor.
    "graph_exaone_tabular_classification.onnx"
    "graph_exaone_tabular_regression.onnx"
    "graph_migraphx_exaone_tabular_classification.onnx"
    "graph_migraphx_exaone_tabular_regression.onnx"
    "tensor_map_exaone_tabular_classification.json"
    "tensor_map_exaone_tabular_regression.json"
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
    # Orion-BiX (MIT) — classification only, no regression graph upstream. The CPU graph is (x, y)-only (split =
    # len(y)); the migraphx graph is the MASKED contract (x, y_full, train_size, d; tools/export_orion_bix
    # --contract mask) so ROCm can bucket it. It reads the same model.safetensors as the ext graph.
    "graph_orion_bix_classification.onnx"
    "graph_migraphx_orion_bix_classification.onnx"
    "tensor_map_orion_bix_classification.json"
    # Orion-MSP (MIT) — classification only, like its Orion-BiX sibling, and like it with a second, MASKED
    # migraphx graph for ROCm (tools/export_orion_msp --contract mask; the frozen random-link table stays inline).
    "graph_orion_msp_classification.onnx"
    "graph_migraphx_orion_msp_classification.onnx"
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
    "tensor_map_tabpfn3_regression.json"
    # TabPFN-3.5 and 3.5-Fast (Prior Labs, non-commercial; tools/export_tabpfn35). Same depth-only
    # difference between the two (24 vs 8 ICL layers), so separate stems. Every kind is bundled: the
    # plain CPU graph, the external-data graph (CUDA + CPU low-memory) and the masked MIGraphX graph.
    "graph_tabpfn35_classification.onnx"
    "graph_tabpfn35_regression.onnx"
    "tensor_map_tabpfn35_classification.json"
    "tensor_map_tabpfn35_regression.json"
    "graph_ext_tabpfn35_classification.onnx"
    "graph_ext_tabpfn35_regression.onnx"
    "graph_migraphx_tabpfn35_classification.onnx"
    "graph_migraphx_tabpfn35_regression.onnx"
    "graph_tabpfn35fast_classification.onnx"
    "graph_tabpfn35fast_regression.onnx"
    "tensor_map_tabpfn35fast_classification.json"
    "tensor_map_tabpfn35fast_regression.json"
    "graph_ext_tabpfn35fast_classification.onnx"
    "graph_ext_tabpfn35fast_regression.onnx"
    "graph_migraphx_tabpfn35fast_classification.onnx"
    "graph_migraphx_tabpfn35fast_regression.onnx"
    # Kumo-Tabular (NVIDIA; code Apache-2.0, weights OpenMDW-1.1), one stem per SIZE (the three are
    # different module trees). The graph takes the tabfm-v1 five-input contract (the split and the
    # real width are VALUES), so the plain, ext and migraphx graphs are one family: plain for the CPU
    # injection path (the published .pt is read natively), ext + migraphx for the converted
    # model.safetensors tabfm_download writes beside the .pt, gated by ExpectedWeightsHeaderShaFor.
    "graph_kumo_small_classification.onnx"
    "graph_kumo_small_regression.onnx"
    "graph_ext_kumo_small_classification.onnx"
    "graph_ext_kumo_small_regression.onnx"
    "graph_migraphx_kumo_small_classification.onnx"
    "graph_migraphx_kumo_small_regression.onnx"
    "tensor_map_kumo_small_classification.json"
    "tensor_map_kumo_small_regression.json"
    "graph_kumo_medium_classification.onnx"
    "graph_kumo_medium_regression.onnx"
    "graph_ext_kumo_medium_classification.onnx"
    "graph_ext_kumo_medium_regression.onnx"
    "graph_migraphx_kumo_medium_classification.onnx"
    "graph_migraphx_kumo_medium_regression.onnx"
    "tensor_map_kumo_medium_classification.json"
    "tensor_map_kumo_medium_regression.json"
    "graph_kumo_large_classification.onnx"
    "graph_kumo_large_regression.onnx"
    "graph_ext_kumo_large_classification.onnx"
    "graph_ext_kumo_large_regression.onnx"
    "graph_migraphx_kumo_large_classification.onnx"
    "graph_migraphx_kumo_large_regression.onnx"
    "tensor_map_kumo_large_classification.json"
    "tensor_map_kumo_large_regression.json"
    # OpenMDW-1.1, verbatim from nvidia/Kumo-Tabular@b2f5a9d6 (LICENSE): the agreement whoever distributes
    # a portion of the Kumo Model Materials must retain. The graphs above may be such a portion, so the
    # binary carries it; tabfm_list_models() shows it (the registry's license.text_resource).
    "license_openmdw_1_1_nvidia.txt")

# Re-run configure when a resource changes so the generated source stays fresh.
foreach(_f ${_tabfm_inputs})
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_tabfm_res_dir}/${_f}")
endforeach()

# Regenerate a file only when it is missing or stale (the hex embedding is a few seconds per MB).
#
# "Stale" includes THIS FILE changing, not only an input being newer. Adding an entry to the list
# above whose resource is OLDER than the generated source (the usual order: export the graph, then
# register it) left the stale source in place, so the new resource silently was not embedded and
# GetBundledResource() returned null. For an ext graph that means the engine quietly takes the
# injection path instead: no error, same answers, wrong path. A fresh CI build never sees it.
set(_tabfm_res_out "${CMAKE_CURRENT_BINARY_DIR}/tabfm_res")
file(MAKE_DIRECTORY "${_tabfm_res_out}")

set(_tabfm_sources "${_tabfm_gen}")
set(_decls "")
set(_table "")
set(_idx 0)
foreach(_f ${_tabfm_inputs})
    set(_path "${_tabfm_res_dir}/${_f}")
    if(NOT EXISTS "${_path}")
        message(FATAL_ERROR "anofox_tabfm: bundled resource missing: ${_path}")
    endif()
    set(_unit "${_tabfm_res_out}/tabfm_res_${_idx}.cpp")
    list(APPEND _tabfm_sources "${_unit}")
    file(SIZE "${_path}" _bytes)

    set(_unit_stale FALSE)
    if(NOT EXISTS "${_unit}")
        set(_unit_stale TRUE)
    elseif("${CMAKE_CURRENT_LIST_FILE}" IS_NEWER_THAN "${_unit}")
        set(_unit_stale TRUE)
    elseif("${_path}" IS_NEWER_THAN "${_unit}")
        set(_unit_stale TRUE)
    endif()
    if(_unit_stale)
        message(STATUS "anofox_tabfm: embedding ${_f} -> ${_unit}")
        file(READ "${_path}" _hex HEX)
        string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _arr "${_hex}")
        # `extern` first: a namespace-scope const has internal linkage otherwise, and the index file
        # (another translation unit) must see it.
        set(_body "// Generated by cmake/embed_resources.cmake from resources/${_f} -- do not edit.\n")
        string(APPEND _body "namespace duckdb {\nnamespace anofox {\n\n")
        string(APPEND _body "extern const unsigned char kTabfmRes${_idx}[];\n")
        string(APPEND _body "const unsigned char kTabfmRes${_idx}[] = {${_arr}};\n\n")
        string(APPEND _body "} // namespace anofox\n} // namespace duckdb\n")
        file(WRITE "${_unit}" "${_body}")
    endif()

    string(APPEND _decls "extern const unsigned char kTabfmRes${_idx}[];\n")
    string(APPEND _table "    {\"${_f}\", kTabfmRes${_idx}, ${_bytes}UL},\n")
    if(_f MATCHES "\\.onnx$")
        string(REGEX REPLACE "\\.onnx$" "" _stem "${_f}")
        string(APPEND _table "    {\"${_stem}\", kTabfmRes${_idx}, ${_bytes}UL},\n")
    endif()
    math(EXPR _idx "${_idx}+1")
endforeach()

# The index. It is small, but it is rewritten whenever the list or this file changes, so a removed or
# added entry is never left half-registered.
set(_index_stale FALSE)
if(NOT EXISTS "${_tabfm_gen}")
    set(_index_stale TRUE)
elseif("${CMAKE_CURRENT_LIST_FILE}" IS_NEWER_THAN "${_tabfm_gen}")
    set(_index_stale TRUE)
else()
    foreach(_f ${_tabfm_inputs})
        if("${_tabfm_res_dir}/${_f}" IS_NEWER_THAN "${_tabfm_gen}")
            set(_index_stale TRUE)
        endif()
    endforeach()
endif()
if(_index_stale)
    message(STATUS "anofox_tabfm: writing resource index -> ${_tabfm_gen}")
    set(_body "// Generated by cmake/embed_resources.cmake -- do not edit.\n")
    string(APPEND _body "#include \"tabfm_bundled_resources.hpp\"\n\n")
    string(APPEND _body "namespace duckdb {\nnamespace anofox {\n\n")
    string(APPEND _body "${_decls}\n")
    string(APPEND _body "namespace {\n\nstruct BundledEntry {\n")
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

set(TABFM_BUNDLED_RESOURCES_SRC ${_tabfm_sources})
