CURRENT_DIR := $(CURDIR)
UNAME_S := $(shell uname -s)

# CMake arguments
CMAKE_ARGS := -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
CMAKE_ARGS += -DBUILD_SHARED_LIBS=$(BUILD_SHARED_LIBS)
CMAKE_ARGS += -DNSPARSE_ENABLE_PYTHON=ON
CMAKE_ARGS += -DPython_EXECUTABLE=$(shell which python)
CMAKE_ARGS += -DCMAKE_BUILD_TYPE=Release
CMAKE_ARGS += -DNSPARSE_OPT_LEVEL=avx512
ifeq ($(UNAME_S),Darwin)
CMAKE_ARGS += -DCMAKE_C_COMPILER=/usr/bin/clang
CMAKE_ARGS += -DCMAKE_CXX_COMPILER=/usr/bin/clang++
CMAKE_ARGS += -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" 
CMAKE_ARGS += -DOpenMP_CXX_LIB_NAMES="omp"
CMAKE_ARGS += -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib
endif
#CMAKE_ARGS += -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
CMAKE_ARGS += -DNSPARSE_ENABLE_TESTS=ON
CMAKE_ARGS += -DNSPARSE_ENABLE_BENCHMARKS=ON

.PHONY: all clean

all: clean
	cmake -B build --fresh $(CMAKE_ARGS)
	cmake --build build
	ln -sf build/compile_commands.json .

clean:
	-rm -rf build
	-rm -rf nsparse/python/build

build: clean
	mkdir -p build
	cmake -S $(CURRENT_DIR) -B build $(CMAKE_ARGS)
	cmake --build build
	ln -sf build/compile_commands.json .

test:
	ctest --test-dir build