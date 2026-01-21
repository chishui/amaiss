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
set_source_files_properties(swigamaiss.swig PROPERTIES
  CPLUSPLUS ON
  USE_TARGET_INCLUDE_DIRECTORIES TRUE
)
set_source_files_properties(swigamaiss.swig PROPERTIES
  SWIG_FLAGS -DSWIGWORDSIZE64
)
set(CMAKE_SWIG_FLAGS -c++ -I${PROJECT_SOURCE_DIR})
set_property(SOURCE swigamaiss.swig PROPERTY CPLUSPLUS ON)
set_property(SOURCE swigamaiss.swig PROPERTY USE_TARGET_INCLUDE_DIRECTORIES ON)

# Create SWIG Python module
swig_add_library(swigamaiss
  TYPE SHARED
  LANGUAGE python
  SOURCES swigamaiss.swig
)

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

