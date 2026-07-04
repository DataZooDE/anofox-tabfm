# ONNX Runtime acquisition for the anofox_tabfm extension.
#
# One codebase, three flavors (HLD D9): the flavor only decides which ONNX
# Runtime build is linked and which execution providers it carries.
#
#   TABFM_FLAVOR=cpu   (default)  ORT CPU EP.
#                                 Source: vcpkg port (manifest feature
#                                 "ort-vcpkg", CI default) or the official
#                                 prebuilt release archive (local default).
#   TABFM_FLAVOR=cuda             Prebuilt ORT-GPU archive (CUDA EP + CPU EP).
#                                 CUDA/cuDNN resolve from the user's system at
#                                 runtime; never linked into this build.
#   TABFM_FLAVOR=rocm             ORT built from source with the MIGraphX EP
#                                 (no official prebuilt exists). Point
#                                 TABFM_ORT_ROCM_DIR at that install tree.
#
# Prebuilt archives default to the official GitHub release URLs but are
# mirror-configurable (S02/S05: GitHub may be blocked; NuGet payloads work
# too) via TABFM_ORT_URL.
#
# Outputs: tabfm_onnxruntime INTERFACE target (includes + libs) and
# TABFM_ORT_PROVIDERS (list of provider compile definitions).

set(TABFM_FLAVOR "cpu" CACHE STRING "anofox_tabfm flavor: cpu | cuda | rocm")
# 1.23.2 matches the ROCm-flavor ORT build; the CPU prebuilt is used by default.
set(TABFM_ORT_VERSION "1.23.2" CACHE STRING "ONNX Runtime version for prebuilt archives")
set(TABFM_ORT_URL "" CACHE STRING "Override URL for the prebuilt ONNX Runtime archive (mirror support)")
set(TABFM_ORT_ROCM_DIR "" CACHE PATH "Install tree of an ONNX Runtime build with --use_migraphx (rocm flavor)")

# IMPORTED so the target may be referenced by the exported extension targets
# (install(EXPORT DuckDBExports) rejects non-imported build-tree targets).
add_library(tabfm_onnxruntime INTERFACE IMPORTED GLOBAL)
set(TABFM_ORT_PROVIDERS "")

# Map host arch to the ORT release archive suffix.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(_ort_platform "linux-aarch64")
    else()
        set(_ort_platform "linux-x64")
    endif()
    set(_ort_ext "tgz")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(_ort_platform "osx-arm64")
    set(_ort_ext "tgz")
elseif(WIN32)
    set(_ort_platform "win-x64")
    set(_ort_ext "zip")
endif()

function(_tabfm_fetch_prebuilt_ort archive_stem)
    if(TABFM_ORT_URL)
        set(_url "${TABFM_ORT_URL}")
    else()
        set(_url "https://github.com/microsoft/onnxruntime/releases/download/v${TABFM_ORT_VERSION}/${archive_stem}-${TABFM_ORT_VERSION}.${_ort_ext}")
    endif()
    include(FetchContent)
    FetchContent_Declare(ort_prebuilt URL "${_url}")
    FetchContent_MakeAvailable(ort_prebuilt)
    set(_ort_root "${ort_prebuilt_SOURCE_DIR}")
    target_include_directories(tabfm_onnxruntime INTERFACE "${_ort_root}/include")
    find_library(TABFM_ORT_LIB onnxruntime PATHS "${_ort_root}/lib" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(tabfm_onnxruntime INTERFACE "${TABFM_ORT_LIB}")
    # The loadable extension must find libonnxruntime.so* next to itself or on
    # the system; record the lib dir for install/packaging steps.
    set(TABFM_ORT_LIB_DIR "${_ort_root}/lib" PARENT_SCOPE)
endfunction()

if(TABFM_FLAVOR STREQUAL "cpu")
    # Prefer a config package (vcpkg port via manifest feature, or system).
    find_package(onnxruntime CONFIG QUIET)
    if(onnxruntime_FOUND)
        message(STATUS "anofox_tabfm: ONNX Runtime from vcpkg/system package (CPU EP)")
        target_link_libraries(tabfm_onnxruntime INTERFACE onnxruntime::onnxruntime)
    else()
        message(STATUS "anofox_tabfm: ONNX Runtime from prebuilt archive v${TABFM_ORT_VERSION} (CPU EP)")
        _tabfm_fetch_prebuilt_ort("onnxruntime-${_ort_platform}")
    endif()
elseif(TABFM_FLAVOR STREQUAL "cuda")
    if(NOT _ort_platform MATCHES "linux-x64|win-x64")
        message(FATAL_ERROR "anofox_tabfm: cuda flavor is only available on linux_amd64/windows_amd64 (got ${_ort_platform})")
    endif()
    message(STATUS "anofox_tabfm: ONNX Runtime GPU (CUDA EP) from prebuilt archive v${TABFM_ORT_VERSION}")
    _tabfm_fetch_prebuilt_ort("onnxruntime-${_ort_platform}-gpu")
    list(APPEND TABFM_ORT_PROVIDERS "TABFM_EP_CUDA=1")
elseif(TABFM_FLAVOR STREQUAL "rocm")
    if(NOT TABFM_ORT_ROCM_DIR)
        message(FATAL_ERROR "anofox_tabfm: rocm flavor needs TABFM_ORT_ROCM_DIR pointing at an ONNX Runtime install built with --use_migraphx (no official prebuilt exists; see docs/rocm-build.md)")
    endif()
    message(STATUS "anofox_tabfm: ONNX Runtime with MIGraphX EP from ${TABFM_ORT_ROCM_DIR}")
    target_include_directories(tabfm_onnxruntime INTERFACE "${TABFM_ORT_ROCM_DIR}/include" "${TABFM_ORT_ROCM_DIR}/include/onnxruntime")
    find_library(TABFM_ORT_LIB onnxruntime PATHS "${TABFM_ORT_ROCM_DIR}/lib" "${TABFM_ORT_ROCM_DIR}/lib64" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(tabfm_onnxruntime INTERFACE "${TABFM_ORT_LIB}")
    set(TABFM_ORT_LIB_DIR "${TABFM_ORT_ROCM_DIR}/lib")
    list(APPEND TABFM_ORT_PROVIDERS "TABFM_EP_MIGRAPHX=1")

    # Direct MIGraphX backend (tabfm_migraphx.cpp): ORT's MIGraphX EP can't handle
    # >2 GB models, so ROCm inference goes straight through MIGraphX. Needs the
    # MIGraphX headers + the C-API lib (migraphx.hpp is a header-only wrapper over
    # libmigraphx_c). Defaults to /opt/rocm; override for a local extracted prefix.
    set(TABFM_MIGRAPHX_DIR "/opt/rocm" CACHE PATH "MIGraphX install prefix (include/ + lib/libmigraphx_c)")
    target_include_directories(tabfm_onnxruntime INTERFACE "${TABFM_MIGRAPHX_DIR}/include")
    find_library(TABFM_MIGRAPHX_LIB migraphx_c PATHS "${TABFM_MIGRAPHX_DIR}/lib" "${TABFM_MIGRAPHX_DIR}/lib64" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(tabfm_onnxruntime INTERFACE "${TABFM_MIGRAPHX_LIB}")
else()
    message(FATAL_ERROR "anofox_tabfm: unknown TABFM_FLAVOR '${TABFM_FLAVOR}' (expected cpu | cuda | rocm)")
endif()

string(TOUPPER "${TABFM_FLAVOR}" _flavor_upper)
target_compile_definitions(tabfm_onnxruntime INTERFACE
    "TABFM_FLAVOR_NAME=\"${TABFM_FLAVOR}\""
    "TABFM_FLAVOR_${_flavor_upper}=1"
    ${TABFM_ORT_PROVIDERS})
