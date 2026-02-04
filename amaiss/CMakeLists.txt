set(AMAISS_SRC
    index.cpp
    brutal_index.cpp
    seismic_index.cpp
    seismic_scalar_quantized_index.cpp
    sparse_vectors.cpp
    index_factory.cpp
    cluster/random_kmeans.cpp
    cluster/kmeans_utils.cpp
    cluster/inverted_list_clusters.cpp
    invlists/inverted_lists.cpp
    io/file_io.cpp
    io/index_io.cpp
)

set(AMAISS_HEADERS
    index.h
    types.h
    sparse_vectors.h
    brutal_index.h
    seismic_index.h
    seismic_scalar_quantized_index.h
    index_factory.h
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
    io/io.h
    io/seismic_invlists_writer.h
    io/file_io.h
    io/index_io.h
)

add_library(amaiss ${AMAISS_SRC} ${AMAISS_HEADERS})

# Detect architecture - x86 SIMD targets only make sense on x86_64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86)")
  set(AMAISS_IS_X86 TRUE)
else()
  set(AMAISS_IS_X86 FALSE)
endif()

# AVX2 target - x86 only
if(AMAISS_IS_X86)
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
else()
  # Create a dummy target that just aliases the base library on non-x86
  add_library(amaiss_avx2 ALIAS amaiss)
endif()

# AVX512 target - x86 only
if(AMAISS_IS_X86)
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
else()
  add_library(amaiss_avx512 ALIAS amaiss)
endif()

# AVX512 Sapphire Rapids target - x86 only
if(AMAISS_IS_X86)
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
else()
  add_library(amaiss_avx512_spr ALIAS amaiss)
endif()

# SVE target - ARM only (but note: Apple Silicon doesn't support SVE)
# Detect ARM architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|ARM64)")
  set(AMAISS_IS_ARM TRUE)
else()
  set(AMAISS_IS_ARM FALSE)
endif()

if(AMAISS_IS_ARM AND NOT APPLE)
  # SVE is available on some ARM processors but NOT on Apple Silicon
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
  set_target_properties(amaiss_sve PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(amaiss_sve PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
endif()

# Set properties for base library (always built)
set_target_properties(amaiss PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(amaiss PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

# Set properties for x86 SIMD targets (only if they're real targets, not aliases)
if(AMAISS_IS_X86)
  set_target_properties(amaiss_avx2 amaiss_avx512 amaiss_avx512_spr PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(amaiss_avx2 PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
  target_include_directories(amaiss_avx512 PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
  target_include_directories(amaiss_avx512_spr PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
endif()


if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()
find_package(OpenMP REQUIRED)

# Link base library (always)
target_link_libraries(amaiss PRIVATE OpenMP::OpenMP_CXX absl::flat_hash_set)

# Link x86 SIMD targets (only if they're real targets)
if(AMAISS_IS_X86)
  target_link_libraries(amaiss_avx2 PRIVATE OpenMP::OpenMP_CXX absl::flat_hash_set)
  target_link_libraries(amaiss_avx512 PRIVATE OpenMP::OpenMP_CXX absl::flat_hash_set)
  target_link_libraries(amaiss_avx512_spr PRIVATE OpenMP::OpenMP_CXX absl::flat_hash_set)
endif()

# Link SVE target (only if it exists as a real target)
if(AMAISS_IS_ARM AND NOT APPLE)
  target_link_libraries(amaiss_sve PRIVATE OpenMP::OpenMP_CXX absl::flat_hash_set)
endif()
