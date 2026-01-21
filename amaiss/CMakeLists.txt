set(AMAISS_SRC
    index.cpp
    brutal_index.cpp
    seismic_index.cpp
    sparse_vectors.cpp
    cluster/random_kmeans.cpp
    cluster/kmeans_utils.cpp
    cluster/inverted_list_clusters.cpp
    invlists/inverted_lists.cpp
)

set(AMAISS_HEADERS
    index.h
    types.h
    sparse_vectors.h
    brutal_index.h
    seismic_index.h
    utils/checks.h
    utils/distance.h
    utils/ranker.h
    cluster/random_kmeans.h
    cluster/inverted_list_clusters.h
    cluster/kmeans_utils.h
    cluster/inverted_list_clusters.h
    invlists/inverted_lists.h
)

add_library(amaiss ${AMAISS_SRC})
set_target_properties(amaiss PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(amaiss PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)


if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()
find_package(OpenMP REQUIRED)
target_link_libraries(amaiss PRIVATE OpenMP::OpenMP_CXX)
