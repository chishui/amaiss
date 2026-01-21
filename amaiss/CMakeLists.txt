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
    utils/distance_avx512.h
    utils/distance_avx2.h
    utils/ranker.h
    cluster/random_kmeans.h
    cluster/inverted_list_clusters.h
    cluster/kmeans_utils.h
    cluster/inverted_list_clusters.h
    invlists/inverted_lists.h
)

add_library(amaiss ${AMAISS_SRC} ${AMAISS_HEADERS})
add_library(amaiss_avx2 ${AMAISS_SRC} ${AMAISS_HEADERS})
if(NOT AMAISS_OPT_LEVEL STREQUAL "avx2" AND NOT AMAISS_OPT_LEVEL STREQUAL "avx512" AND NOT AMAISS_OPT_LEVEL STREQUAL "avx512_spr")
  set_target_properties(amaiss_avx2 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  target_compile_options(amaiss_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mpopcnt>)
else()
  target_compile_options(amaiss_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
  add_compile_options(/bigobj)
endif()

add_library(amaiss_avx512 ${AMAISS_SRC} ${AMAISS_HEADERS})
if(NOT AMAISS_OPT_LEVEL STREQUAL "avx512")
  set_target_properties(amaiss_avx512 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  target_compile_options(amaiss_avx512 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt>)
else()
  target_compile_options(amaiss_avx512 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
  add_compile_options(/bigobj)
endif()

add_library(amaiss_avx512_spr ${AMAISS_SRC} ${AMAISS_HEADERS})
if(NOT AMAISS_OPT_LEVEL STREQUAL "avx512_spr")
  set_target_properties(amaiss_avx512_spr PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  target_compile_options(amaiss_avx512_spr PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=sapphirerapids -mtune=sapphirerapids>)
else()
  target_compile_options(amaiss_avx512_spr PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
  add_compile_options(/bigobj)
endif()

add_library(amaiss_sve ${AMAISS_SRC} ${AMAISS_HEADERS})
if(NOT AMAISS_OPT_LEVEL STREQUAL "sve")
  set_target_properties(amaiss_sve PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=native")
  elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
    target_compile_options(amaiss_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:${CMAKE_MATCH_2}+sve>)
  elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=armv")
    target_compile_options(amaiss_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-march=armv8-a+sve>)
  endif()
  if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=native")
  elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
    target_compile_options(amaiss_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:${CMAKE_MATCH_2}+sve>)
  elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=armv")
    target_compile_options(amaiss_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-march=armv8-a+sve>)
  endif()
endif()

set_target_properties(amaiss amaiss_avx2 amaiss_avx512 amaiss_avx512_spr amaiss_sve PROPERTIES
POSITION_INDEPENDENT_CODE ON)

target_include_directories(amaiss PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_include_directories(amaiss_avx2 PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_include_directories(amaiss_avx512 PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_include_directories(amaiss_avx512_spr PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_include_directories(amaiss_sve PUBLIC
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
target_link_libraries(amaiss_avx2 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(amaiss_avx512 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(amaiss_avx512_spr PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(amaiss_sve PRIVATE OpenMP::OpenMP_CXX)
