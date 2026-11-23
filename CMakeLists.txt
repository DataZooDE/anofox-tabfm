cmake_minimum_required(VERSION 3.10...3.29)

set(TARGET_NAME anofox_tabfm)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME} LANGUAGES C CXX)
# Feedback banner (header-only). Telemetry ON so a printed banner counts as a
# banner_shown event through the instance configured below.
set(DATAZOO_BANNER_TELEMETRY ON CACHE BOOL "" FORCE)
set(DATAZOO_BANNER_TELEMETRY_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/include CACHE PATH "" FORCE)
add_subdirectory(datazoo-banner)

include_directories(src/include)
# DuckDB 1.5 / 2.0 API differences (header-only). Include path only: a
# target_link_libraries() to duckdb_compat breaks DuckDB's export() of the
# extension targets.
include_directories(duckdb-compat/include)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies ---

# --- ggml engine (docs/GGML_PLAN.md) ---
#
# The CPU device runs on the ggml engine shared with anofox-decide: vendor/anofox-ggml-adapter (a git submodule with ggml as its own
# submodule; a checkout without them does not configure). The adapter's module builds ggml as static archives with hidden visibility
# and gives the engine sources and the anofox_ggml link target. GPUs are served by plugins (tools/ggml_plugin for CUDA and ROCm, the
# MLX plugin below for Apple); nothing links ONNX Runtime. ANOFOX_GGML_NATIVE compiles ggml for this machine's CPU (default OFF).
# The option's name before the adapter, honoured only when the new one is not set, and dropped so a cached value cannot override it.
if(DEFINED TABFM_GGML_NATIVE AND NOT DEFINED CACHE{ANOFOX_GGML_NATIVE})
    set(ANOFOX_GGML_NATIVE ${TABFM_GGML_NATIVE} CACHE BOOL "Compile ggml for this machine's CPU")
endif()
unset(TABFM_GGML_NATIVE CACHE)
include(vendor/anofox-ggml-adapter/cmake/anofox_ggml_adapter.cmake)

# Embed the weight-free graphs + tensor maps from resources/ into the binary
# (built-in model flow needs no companion files) — sets
# TABFM_BUNDLED_RESOURCES_SRC to the generated source.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_resources.cmake)

# --- Telemetry ---
# Build telemetry as OBJECT library to avoid link order issues with DuckDB symbols.
find_package(OpenSSL REQUIRED)

add_library(posthog_telemetry_obj OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/src/telemetry.cpp
)

target_include_directories(posthog_telemetry_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/include
    ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/third_party/httplib
    ${OPENSSL_INCLUDE_DIR}
)

target_compile_features(posthog_telemetry_obj PUBLIC cxx_std_17)

# --- Build ---

set(EXTENSION_SOURCES
        src/anofox_tabfm_extension.cpp
        src/tabfm_settings.cpp
        src/tabfm_cpu_budget.cpp
        src/tabfm_state.cpp
        src/tabfm_manifest.cpp
        src/tabfm_model_spec.cpp
        src/tabfm_registry.cpp
        src/tabfm_ckpt.cpp
        src/tabfm_safetensors.cpp
        src/tabfm_session.cpp
        src/tabfm_devices.cpp
        src/tabfm_preprocess.cpp
        src/tabfm_ensemble.cpp
        src/tabfm_weights.cpp
        src/tabfm_predict_agg.cpp
        src/tabfm_macros.cpp
        src/tabfm_generate.cpp
        src/tabfm_generate_macros.cpp
        src/tabfm_engine.cpp
        src/tabfm_plugin_backend.cpp
        src/tabfm_metrics_classification.cpp
        src/tabfm_contract.cpp
        src/tabfm_metrics_regression.cpp
        src/tabfm_crossval.cpp
        src/tabfm_scoring.cpp
        ${TABFM_BUNDLED_RESOURCES_SRC}
)
list(APPEND EXTENSION_SOURCES src/tabfm_ggml_session.cpp ${ANOFOX_GGML_ADAPTER_SOURCES})

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Telemetry objects + include path into both extension targets.
foreach(_tgt ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
    target_sources(${_tgt} PRIVATE $<TARGET_OBJECTS:posthog_telemetry_obj>)
    target_include_directories(${_tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/include)
    add_dependencies(${_tgt} ggml)
    target_link_libraries(${_tgt} anofox_ggml)
    target_link_libraries(${_tgt}
        datazoo_banner
        OpenSSL::SSL
        OpenSSL::Crypto)
endforeach()

# posthog-telemetry reads the machine id from the IORegistry on macOS; DuckDB 2.0's httplib loads
# system certificates through the Security framework.
if(APPLE)
    target_link_libraries(${EXTENSION_NAME} "-framework IOKit" "-framework CoreFoundation" "-framework Security")
    target_link_libraries(${LOADABLE_EXTENSION_NAME} "-framework IOKit" "-framework CoreFoundation" "-framework Security")
endif()

# posthog-telemetry's machine id (GetMacAddress -> GetAdaptersInfo) needs the
# Windows IP Helper API.
if(WIN32)
    target_link_libraries(${EXTENSION_NAME} iphlpapi ws2_32)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} iphlpapi ws2_32)
endif()

# GCC 14 COMDAT vs non-COMDAT conflict on Linux for DuckDB v1.5+ (see
# extension_config.cmake for the full explanation).
if(UNIX AND NOT APPLE)
    target_link_options(${EXTENSION_NAME} PUBLIC -Wl,--allow-multiple-definition)
    target_link_options(${LOADABLE_EXTENSION_NAME} PRIVATE -Wl,--allow-multiple-definition)
endif()

# --- Apple MLX backend plugin (docs/MLX_PLAN.md, spikes in docs/MLX_SPIKE_RESULTS.md) ---
#
# A plugin the extension dlopens, like the ggml GPU plugin (tools/ggml_plugin). MLX has no ONNX importer, so the
# plugin runs the model's ONNX graph through its own interpreter (src/tabfm_mlx_graph.cpp + the adapter's ONNX reader);
# mitra also has a forward transcribed from mitra_model_patched.py, kept as its fast path.
#
# Optional and silent when absent, like the others: Linux and Intel-Mac CI
# runners carry no MLX and must not fail configure over a plugin they cannot
# use. Guarded on APPLE + arm64 as well, since MLX exists only there.
set(TABFM_MLX_DIR "/opt/homebrew" CACHE PATH "MLX + mlx-c install prefix (include/mlx/c + lib/libmlxc) for the plugin build")
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    find_library(TABFM_MLX_C_LIB mlxc PATHS "${TABFM_MLX_DIR}/lib" NO_DEFAULT_PATH)
    find_library(TABFM_MLX_LIB mlx PATHS "${TABFM_MLX_DIR}/lib" NO_DEFAULT_PATH)
    if(TABFM_MLX_C_LIB AND TABFM_MLX_LIB AND EXISTS "${TABFM_MLX_DIR}/include/mlx/c/ops.h")
        message(STATUS "anofox_tabfm: building the mlx backend plugin from ${TABFM_MLX_DIR}")
        # The plugin carries its own minimal ONNX reader rather than linking
        # libonnx + libprotobuf: it is separately distributed, and coupling a
        # small .dylib to protobuf's ABI (and to whatever abseil the extension
        # built against) buys nothing it needs. See the adapter's onnx_reader.hpp.
        add_library(anofox_tabfm_mlx_plugin SHARED
            src/tabfm_mlx_plugin.cpp
            src/tabfm_mlx_graph.cpp
            vendor/anofox-ggml-adapter/src/onnx_reader.cpp)
        target_include_directories(anofox_tabfm_mlx_plugin PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/include
            "${ANOFOX_GGML_ADAPTER_INCLUDE_DIR}"
            "${TABFM_MLX_DIR}/include")
        target_link_libraries(anofox_tabfm_mlx_plugin PRIVATE "${TABFM_MLX_C_LIB}" "${TABFM_MLX_LIB}")
        target_compile_features(anofox_tabfm_mlx_plugin PRIVATE cxx_std_17)
        # mlx-c 0.7 changed two calls the plugin makes: cumsum over an axis became mlx_cumsum_axis with an
        # output dtype, and fast_scaled_dot_product_attention gained force_fused. mlx-c has no version macro,
        # so probe for the new API and let the sources pick the matching signature.
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_INCLUDES "${TABFM_MLX_DIR}/include")
        set(CMAKE_REQUIRED_LIBRARIES "${TABFM_MLX_C_LIB}" "${TABFM_MLX_LIB}")
        check_cxx_source_compiles("
#include \"mlx/c/ops.h\"
int main() {
    mlx_array r = mlx_array_new();
    mlx_optional_dtype none;
    none.has_value = false;
    return mlx_cumsum_axis(&r, r, 0, false, true, none, mlx_default_cpu_stream_new());
}" TABFM_MLX_C_HAS_CUMSUM_AXIS)
        unset(CMAKE_REQUIRED_INCLUDES)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(TABFM_MLX_C_HAS_CUMSUM_AXIS)
            target_compile_definitions(anofox_tabfm_mlx_plugin PRIVATE TABFM_MLX_C_07=1)
        endif()
        set_target_properties(anofox_tabfm_mlx_plugin PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set(TABFM_MLX_PLUGIN_BUILT TRUE)
    else()
        message(STATUS "anofox_tabfm: no MLX/mlx-c install found at ${TABFM_MLX_DIR}, skipping the mlx backend plugin")
    endif()
endif()

# --- C++ unit tests (Catch2, compiled into DuckDB's test/unittest binary) ---
#
# Registered through a deferred call because this CMakeLists is processed via
# add_subdirectory() from the DuckDB build before duckdb/test/ creates the
# `unittest` target. OBJECT library so Catch2 auto-registration TUs are linked
# into the binary directly.
set(TABFM_CPP_TEST_SOURCES
    test/cpp/test_tabfm_scaffold.cpp
    test/cpp/test_tabfm_safetensors.cpp
    test/cpp/test_tabfm_manifest.cpp
    test/cpp/test_tabfm_session.cpp
    test/cpp/test_tabfm_model_spec.cpp
    test/cpp/test_tabfm_registry.cpp
    test/cpp/test_tabfm_ckpt.cpp
    test/cpp/test_tabfm_preprocess.cpp
    test/cpp/test_tabfm_ensemble.cpp
    test/cpp/test_tabfm_generate.cpp
    test/cpp/test_tabfm_bundled_resources.cpp
    test/cpp/test_tabfm_weights.cpp
    test/cpp/test_tabfm_plugin_backend.cpp
    test/cpp/test_tabfm_mlx_plugin.cpp
    test/cpp/test_tabfm_mlx_ops.cpp
    test/cpp/test_tabfm_owned_registry.cpp
    test/cpp/test_tabfm_cpu_budget.cpp
    test/cpp/test_tabfm_metrics.cpp
    test/cpp/test_tabfm_metrics_regression.cpp
    test/cpp/test_tabfm_crossval.cpp
    test/cpp/test_tabfm_scoring.cpp
    test/cpp/test_tabfm_context_cache.cpp
)
list(APPEND TABFM_CPP_TEST_SOURCES test/cpp/test_tabfm_ggml_lowering.cpp test/cpp/test_tabfm_ggml_nori.cpp test/cpp/test_tabfm_ggml_graphs.cpp test/cpp/test_tabfm_shape_eval.cpp)
if(NOT CLANG_TIDY AND NOT EMSCRIPTEN AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
    # Backend-plugin fixtures. The real GPU
    # backend needs hardware CI does not have, which would leave the LOADER —
    # where an ABI mismatch is undefined behaviour rather than a wrong answer —
    # untested. These stand in for it: real shared libraries, one correct and
    # one built with a deliberately wrong ABI version.
    add_library(anofox_tabfm_fake_plugin SHARED test/cpp/plugin_fixture/fake_plugin.cpp)
    target_include_directories(anofox_tabfm_fake_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
    target_compile_features(anofox_tabfm_fake_plugin PRIVATE cxx_std_17)
    set_target_properties(anofox_tabfm_fake_plugin PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(anofox_tabfm_fake_plugin_bad_abi SHARED test/cpp/plugin_fixture/fake_plugin.cpp)
    target_include_directories(anofox_tabfm_fake_plugin_bad_abi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
    target_compile_features(anofox_tabfm_fake_plugin_bad_abi PRIVATE cxx_std_17)
    target_compile_definitions(anofox_tabfm_fake_plugin_bad_abi PRIVATE TABFM_FAKE_PLUGIN_BAD_ABI=1)
    set_target_properties(anofox_tabfm_fake_plugin_bad_abi PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # src/tabfm_ensemble.cpp is not called from the extension yet, only from its
    # tests. DuckDB 2.0 links the extension into unittest through libduckdb.so,
    # which drops archive members nothing references, so the tests compile it
    # themselves. (On 1.x the archive member then simply goes unused.)
    add_library(anofox_tabfm_unittest_tests OBJECT ${TABFM_CPP_TEST_SOURCES} src/tabfm_ensemble.cpp)
    add_dependencies(anofox_tabfm_unittest_tests anofox_tabfm_fake_plugin anofox_tabfm_fake_plugin_bad_abi)
    target_compile_definitions(anofox_tabfm_unittest_tests PRIVATE
        TABFM_FAKE_PLUGIN_PATH="$<TARGET_FILE:anofox_tabfm_fake_plugin>"
        TABFM_FAKE_PLUGIN_BAD_ABI_PATH="$<TARGET_FILE:anofox_tabfm_fake_plugin_bad_abi>")
    if(TABFM_MLX_PLUGIN_BUILT)
        # Apple MLX (docs/MLX_PLAN.md): only where the plugin was built, i.e.
        # an arm64 Mac with MLX installed. The equivalence case additionally
        # needs the developer's mitra cache and skips itself without it; the
        # refusal case runs anywhere the plugin exists.
        add_dependencies(anofox_tabfm_unittest_tests anofox_tabfm_mlx_plugin)
        target_compile_definitions(anofox_tabfm_unittest_tests PRIVATE
            TABFM_MLX_PLUGIN_PATH="$<TARGET_FILE:anofox_tabfm_mlx_plugin>")
    endif()
    target_compile_features(anofox_tabfm_unittest_tests PRIVATE cxx_std_17)
    target_include_directories(anofox_tabfm_unittest_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/third_party/catch
    )
    target_link_libraries(anofox_tabfm_unittest_tests PRIVATE duckdb_static)
    add_dependencies(anofox_tabfm_unittest_tests ggml)
    target_link_libraries(anofox_tabfm_unittest_tests PRIVATE anofox_ggml)

    function(anofox_tabfm_register_cpp_tests)
        if(TARGET unittest)
            target_sources(unittest PRIVATE $<TARGET_OBJECTS:anofox_tabfm_unittest_tests>)
        endif()
    endfunction()
    cmake_language(DEFER DIRECTORY ${CMAKE_SOURCE_DIR} CALL anofox_tabfm_register_cpp_tests)
endif()

# --- Installation ---

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

install(
  TARGETS ${LOADABLE_EXTENSION_NAME}
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
