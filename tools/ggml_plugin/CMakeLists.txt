# The ggml GPU plugin (src/tabfm_ggml_plugin.cpp): a standalone shared library the extension dlopens, like the ORT-CUDA and
# MIGraphX plugins, but running the SAME ggml engine as the CPU build on a ggml GPU device.
#
#   cmake -S tools/ggml_plugin -B build/ggml-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DTABFM_GGML_GPU=cuda -DCMAKE_CUDA_ARCHITECTURES=89
#   cmake -S tools/ggml_plugin -B build/ggml-rocm -G Ninja -DCMAKE_BUILD_TYPE=Release -DTABFM_GGML_GPU=hip  -DAMDGPU_TARGETS=gfx1201
#   cmake --build build/ggml-cuda
#
# The output is libanofox_tabfm_ggml-cuda_plugin.so / libanofox_tabfm_ggml-rocm_plugin.so (no 'lib' and a .dll on Windows). Put it in the
# directory anofox_tabfm_ep_path names: the host prefers it over the ORT-CUDA / MIGraphX plugin when it is there (TryGgmlGpuBackend).
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
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)      # libgomp's spin-waiting starves ggml's own thread pool (docs/GGML_PLAN.md)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)      # a distributable plugin must not depend on the build machine's CPU
set(CMAKE_POSITION_INDEPENDENT_CODE ON)        # ggml is static and goes into a shared library
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${_root}/vendor/ggml" ggml EXCLUDE_FROM_ALL)
# NaN is valid data for these graphs; ggml's CPU kernels assert !isnan in a debug build.
foreach(_t ggml ggml-base ggml-cpu)
    target_compile_definitions(${_t} PRIVATE NDEBUG)
endforeach()

add_library(anofox_tabfm_ggml_plugin SHARED
    "${_root}/src/tabfm_ggml_plugin.cpp"
    "${_root}/src/tabfm_ggml_lower.cpp"
    "${_root}/src/tabfm_ggml_engine.cpp"
    "${_root}/src/tabfm_shape_eval.cpp"
    "${_root}/src/tabfm_onnx_reader.cpp")
target_include_directories(anofox_tabfm_ggml_plugin PRIVATE "${_root}/src/include")
target_compile_features(anofox_tabfm_ggml_plugin PRIVATE cxx_std_17)
set_target_properties(anofox_tabfm_ggml_plugin PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    OUTPUT_NAME "anofox_tabfm_${_plugin_backend}_plugin")
if(WIN32)
    set_target_properties(anofox_tabfm_ggml_plugin PROPERTIES PREFIX "")
endif()
target_link_libraries(anofox_tabfm_ggml_plugin PRIVATE ggml)
