# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

project(amaiss
  VERSION 0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Third party dependencies
include(cmake/third_party.cmake)

# Valid values are "generic", "avx2", "avx512", "sve".
option(AMAISS_OPT_LEVEL "" "generic")
option(AMAISS_ENABLE_PYTHON "Build Python bindings" OFF)

add_subdirectory(amaiss)

# Python bindings
if(AMAISS_ENABLE_PYTHON)
    add_subdirectory(amaiss/python)
endif()

# CTest must be included in the top level to enable `make test` target.
include(CTest)

# Unit tests
option(AMAISS_ENABLE_TESTS "Build unit tests" OFF)
if(AMAISS_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
