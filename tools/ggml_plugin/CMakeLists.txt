# The ggml GPU plugin (src/tabfm_ggml_plugin.cpp): the only CUDA/ROCm backend, a standalone shared library the extension
# dlopens, running the SAME ggml engine as the CPU build (vendor/anofox-ggml-adapter) on a ggml GPU device.
#
#   cmake -S tools/ggml_plugin -B build/ggml-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DTABFM_GGML_GPU=cuda -DCMAKE_CUDA_ARCHITECTURES=89
#   cmake -S tools/ggml_plugin -B build/ggml-rocm -G Ninja -DCMAKE_BUILD_TYPE=Release -DTABFM_GGML_GPU=hip  -DAMDGPU_TARGETS=gfx1201
#   cmake --build build/ggml-cuda
#
# The output is libanofox_tabfm_ggml-cuda_plugin.so / libanofox_tabfm_ggml-rocm_plugin.so (no 'lib' and a .dll on Windows). Put it in the
# directory anofox_tabfm_ep_path names (tabfm_download_runtime('cuda'|'rocm') puts it there): it is the only plugin for CUDA and ROCm (TryGgmlGpuBackend).
cmake_minimum_required(VERSION 3.18)
project(anofox_tabfm_ggml_plugin CXX C)

set(TABFM_GGML_GPU "cuda" CACHE STRING "ggml GPU backend of this plugin: cuda | hip")
set_property(CACHE TABFM_GGML_GPU PROPERTY STRINGS cuda hip)
set(_root "${CMAKE_CURRENT_LIST_DIR}/../..")

if(TABFM_GGML_GPU STREQUAL "cuda")
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(_plugin_backend "ggml-cuda")
elseif(TABFM_GGML_GPU STREQUAL "hip")
    set(GGML_HIP ON CACHE BOOL "" FORCE)
    set(_plugin_backend "ggml-rocm")
else()
    message(FATAL_ERROR "TABFM_GGML_GPU must be cuda or hip")
endif()
set(TABFM_GGML_CUDA_STATIC ON CACHE BOOL "Link the CUDA runtime and cuBLAS into the CUDA plugin statically (only the NVIDIA driver is then needed on the user's machine)")
# The engine and ggml come from the shared adapter (vendor/anofox-ggml-adapter), configured with the GPU backend chosen above: static, no
# OpenMP (libgomp's spin-waiting starves ggml's own pool), llamafile on, GGML_NATIVE off (a distributable plugin must not depend on the build
# machine's CPU), hidden visibility, NDEBUG on ggml (NaN is valid data for these graphs).
set(ANOFOX_GGML_CPU_ONLY OFF)
set(ANOFOX_GGML_NATIVE OFF CACHE BOOL "" FORCE)
include("${_root}/vendor/anofox-ggml-adapter/cmake/anofox_ggml_adapter.cmake")
if(TABFM_GGML_GPU STREQUAL "cuda" AND TABFM_GGML_CUDA_STATIC AND NOT WIN32)
    # ggml's own GGML_STATIC adds -static to every link, which a shared plugin cannot take; select the static CUDA targets for ggml-cuda instead (Windows has no
    # static cuBLAS). The libraries travel as $<LINK_ONLY:CUDA::...> in the static archive's interface, so rewrite both properties. Checked by scripts/check_plugin_deps.sh.
    find_package(CUDAToolkit REQUIRED)
    foreach(_prop LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
        get_target_property(_libs ggml-cuda ${_prop})
        if(_libs)
            string(REGEX REPLACE "CUDA::(cudart|cublasLt|cublas)([>;]|$)" "CUDA::\\1_static\\2" _libs "${_libs}")
            set_property(TARGET ggml-cuda PROPERTY ${_prop} "${_libs}")
        endif()
    endforeach()
endif()

add_library(anofox_tabfm_ggml_plugin SHARED "${_root}/src/tabfm_ggml_plugin.cpp" ${ANOFOX_GGML_ADAPTER_SOURCES})
target_include_directories(anofox_tabfm_ggml_plugin PRIVATE "${_root}/src/include" "${ANOFOX_GGML_ADAPTER_INCLUDE_DIR}")
target_compile_features(anofox_tabfm_ggml_plugin PRIVATE cxx_std_17)
set_target_properties(anofox_tabfm_ggml_plugin PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    OUTPUT_NAME "anofox_tabfm_${_plugin_backend}_plugin")
if(WIN32)
    set_target_properties(anofox_tabfm_ggml_plugin PROPERTIES PREFIX "")
endif()
target_link_libraries(anofox_tabfm_ggml_plugin PRIVATE ggml)
