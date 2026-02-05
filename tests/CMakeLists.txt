# Copyright (c) amaiss authors
# Unit tests for amaiss library

set(AMAISS_TEST_SRC
    distance_test.cpp
    id_map_index_test.cpp
    index_factory_test.cpp
    index_io_test.cpp
    inverted_list_clusters_test.cpp
    inverted_lists_test.cpp
    kmeans_utils_test.cpp
    prefetch_test.cpp
    random_kmeans_test.cpp
    ranker_test.cpp
    scalar_quantizer_test.cpp
    seismic_index_test.cpp
    seismic_invlists_writer_test.cpp
    seismic_scalar_quantized_index_test.cpp
    sparse_vectors_test.cpp
    vector_process_test.cpp
)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(OpenMP REQUIRED)

add_executable(amaiss_test ${AMAISS_TEST_SRC})

target_include_directories(amaiss_test PRIVATE ${PROJECT_SOURCE_DIR})

target_link_libraries(amaiss_test PRIVATE
    amaiss
    OpenMP::OpenMP_CXX
    GTest::gtest_main
    absl::flat_hash_set
)

include(GoogleTest)
gtest_discover_tests(amaiss_test)
