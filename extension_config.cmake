# This file is included by DuckDB's build system. It specifies which extension to load

# GCC 14+ / DuckDB v1.5.x: DuckDB tools that link both duckdb_static (non-COMDAT
# constexpr defs, C++11) and our extension objects (COMDAT defs, C++17) hit a
# "multiple definition" error. Allow it on Linux (harmless ODR quirk).
if(UNIX AND NOT APPLE)
    list(APPEND DUCKDB_EXTRA_LINK_FLAGS -Wl,--allow-multiple-definition)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-multiple-definition")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-multiple-definition")
endif()

duckdb_extension_load(anofox_tabfm
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS)
