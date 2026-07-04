cmake_minimum_required(VERSION 3.10...3.29)

set(TARGET_NAME anofox_tabfm)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME} LANGUAGES C CXX)
include_directories(src/include)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies ---

# ONNX Runtime, flavor-aware (cpu | cuda | rocm) — defines tabfm_onnxruntime.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ort.cmake)

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
        src/tabfm_state.cpp
        src/tabfm_manifest.cpp
        src/tabfm_safetensors.cpp
        src/tabfm_ort_engine.cpp
        src/tabfm_devices.cpp
        src/tabfm_preprocess.cpp
        src/tabfm_ensemble.cpp
        src/tabfm_weights.cpp
        src/tabfm_predict_agg.cpp
        src/tabfm_macros.cpp
)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Telemetry objects + include path into both extension targets.
foreach(_tgt ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
    target_sources(${_tgt} PRIVATE $<TARGET_OBJECTS:posthog_telemetry_obj>)
    target_include_directories(${_tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/include)
    target_link_libraries(${_tgt}
        tabfm_onnxruntime
        OpenSSL::SSL
        OpenSSL::Crypto)
endforeach()

# posthog-telemetry reads the machine id from the IORegistry on macOS.
if(APPLE)
    target_link_libraries(${EXTENSION_NAME} "-framework IOKit" "-framework CoreFoundation")
    target_link_libraries(${LOADABLE_EXTENSION_NAME} "-framework IOKit" "-framework CoreFoundation")
endif()

# GCC 14 COMDAT vs non-COMDAT conflict on Linux for DuckDB v1.5+ (see
# extension_config.cmake for the full explanation).
if(UNIX AND NOT APPLE)
    target_link_options(${EXTENSION_NAME} PUBLIC -Wl,--allow-multiple-definition)
    target_link_options(${LOADABLE_EXTENSION_NAME} PRIVATE -Wl,--allow-multiple-definition)
endif()

# --- C++ unit tests (Catch2, compiled into DuckDB's test/unittest binary) ---
#
# Registered through a deferred call because this CMakeLists is processed via
# add_subdirectory() from the DuckDB build before duckdb/test/ creates the
# `unittest` target. OBJECT library so Catch2 auto-registration TUs are linked
# into the binary directly.
set(TABFM_CPP_TEST_SOURCES
    test/cpp/test_tabfm_scaffold.cpp
)
if(NOT CLANG_TIDY AND NOT EMSCRIPTEN AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
    add_library(anofox_tabfm_unittest_tests OBJECT ${TABFM_CPP_TEST_SOURCES})
    target_compile_features(anofox_tabfm_unittest_tests PRIVATE cxx_std_17)
    target_include_directories(anofox_tabfm_unittest_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/third_party/catch
    )
    target_link_libraries(anofox_tabfm_unittest_tests PRIVATE duckdb_static tabfm_onnxruntime)

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
