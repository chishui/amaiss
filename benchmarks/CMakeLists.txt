# Copyright (c) amaiss authors
# Benchmarks for amaiss library
set(AMAISS_BENCHMARK_SRC
    index_search_benchmark.cpp
)
# Fetch Google Benchmark
include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

# Enable libpfm for perf counters on Linux (requires libpfm4-dev)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(BENCHMARK_ENABLE_LIBPFM ON CACHE BOOL "" FORCE)
endif()

FetchContent_MakeAvailable(googlebenchmark)

if(AMAISS_BENCHMARK_SRC)
    find_package(OpenMP REQUIRED)

    add_executable(amaiss_benchmark ${AMAISS_BENCHMARK_SRC})

    target_include_directories(amaiss_benchmark PRIVATE ${PROJECT_SOURCE_DIR})

    target_link_libraries(amaiss_benchmark PRIVATE
        amaiss
        OpenMP::OpenMP_CXX
        benchmark::benchmark
        benchmark::benchmark_main
        absl::flat_hash_set
        absl::flat_hash_map
    )
endif()
