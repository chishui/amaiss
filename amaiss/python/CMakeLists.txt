# Python bindings for AMAISS

project(pyamaiss
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Find Python first
find_package(Python REQUIRED
  COMPONENTS Interpreter Development.Module NumPy
)

find_package(SWIG REQUIRED COMPONENTS python)
include(${SWIG_USE_FILE})

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(SWIG_SOURCE_FILE_EXTENSIONS swig)

# Configure SWIG
macro(configure_swigamaiss source)
  set_source_files_properties(${source} PROPERTIES
    CPLUSPLUS ON
    USE_TARGET_INCLUDE_DIRECTORIES TRUE
  )
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWORDSIZE64
    )
  endif()
  if(WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWIN
    )
  endif()
endmacro()

configure_file(swigamaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigamaiss_avx2.swig COPYONLY)
configure_file(swigamaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigamaiss_avx512.swig COPYONLY)
configure_file(swigamaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigamaiss_avx512_spr.swig COPYONLY)
configure_file(swigamaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigamaiss_sve.swig COPYONLY)

configure_swigamaiss(swigamaiss.swig)
configure_swigamaiss(swigamaiss_avx2.swig)
configure_swigamaiss(swigamaiss_avx512.swig)
configure_swigamaiss(swigamaiss_avx512_spr.swig)
configure_swigamaiss(swigamaiss_sve.swig)


set(CMAKE_SWIG_FLAGS -c++ -I${PROJECT_SOURCE_DIR})
set_property(SOURCE swigamaiss.swig PROPERTY CPLUSPLUS ON)
set_property(SOURCE swigamaiss.swig PROPERTY USE_TARGET_INCLUDE_DIRECTORIES ON)

# Create SWIG Python module
swig_add_library(swigamaiss
  TYPE SHARED
  LANGUAGE python
  SOURCES swigamaiss.swig
)


set_property(TARGET swigamaiss PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

# Macro to add a SWIG module with optional exclusion based on optimization level
macro(add_swig_module MODULE_NAME OPT_LEVELS)
  set_property(SOURCE swig${MODULE_NAME}.swig
    PROPERTY SWIG_MODULE_NAME swig${MODULE_NAME})
  swig_add_library(swig${MODULE_NAME}
    TYPE SHARED
    LANGUAGE python
    SOURCES swig${MODULE_NAME}.swig
  )
  set_property(TARGET swig${MODULE_NAME} PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

  set(_exclude TRUE)
  foreach(_level ${OPT_LEVELS})
    if(AMAISS_OPT_LEVEL STREQUAL "${_level}")
      set(_exclude FALSE)
      break()
    endif()
  endforeach()

  if(_exclude)
    set_target_properties(swig${MODULE_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
endmacro()

add_swig_module(amaiss_avx2 "avx2;avx512;avx512_spr")
add_swig_module(amaiss_avx512 "avx512")
add_swig_module(amaiss_avx512_spr "avx512_spr")
add_swig_module(amaiss_sve "sve")

if(NOT WIN32)
  # NOTE: Python does not recognize the dylib extension.
  set_target_properties(swigamaiss PROPERTIES SUFFIX .so)
  set_target_properties(swigamaiss_avx2 PROPERTIES SUFFIX .so)
  set_target_properties(swigamaiss_avx512 PROPERTIES SUFFIX .so)
  set_target_properties(swigamaiss_avx512_spr PROPERTIES SUFFIX .so)
  set_target_properties(swigamaiss_sve PROPERTIES SUFFIX .so)
endif()

target_include_directories(swigamaiss PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${NUMPY_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
)

target_link_libraries(swigamaiss PRIVATE
  amaiss
  Python::Module
  Python::NumPy
)

target_link_libraries(swigamaiss_avx2 PRIVATE
  amaiss_avx2
  Python::Module
  Python::NumPy
)

target_link_libraries(swigamaiss_avx512 PRIVATE
  amaiss_avx512
  Python::Module
  Python::NumPy
)

target_link_libraries(swigamaiss_avx512_spr PRIVATE
  amaiss_avx512_spr
  Python::Module
  Python::NumPy
)

target_link_libraries(swigamaiss_sve PRIVATE
  amaiss_sve
  Python::Module
  Python::NumPy
)

# Set output properties
set_target_properties(swigamaiss PROPERTIES
    OUTPUT_NAME "swigamaiss"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON
)

# Copy Python package files to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml
    ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${CMAKE_CURRENT_BINARY_DIR}/__init__.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/loader.py
    ${CMAKE_CURRENT_BINARY_DIR}/loader.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/class_wrappers.py
    ${CMAKE_CURRENT_BINARY_DIR}/class_wrappers.py COPYONLY)

