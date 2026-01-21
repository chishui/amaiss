CURRENT_DIR := $(CURDIR)
UNAME_S := $(shell uname -s)

# CMake arguments
CMAKE_ARGS := -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
CMAKE_ARGS += -DBUILD_SHARED_LIBS=$(BUILD_SHARED_LIBS)
CMAKE_ARGS += -DAMAISS_ENABLE_PYTHON=ON
CMAKE_ARGS += -DPython_EXECUTABLE=$(shell which python)
CMAKE_ARGS += -DCMAKE_BUILD_TYPE=Release

.PHONY: all clean

all: clean
	cmake -B build --fresh $(CMAKE_ARGS)
	cmake --build build
	ln -sf build/compile_commands.json .

clean:
	-rm -rf build

build: clean
	mkdir -p build
	cmake -S $(CURRENT_DIR) -B build $(CMAKE_ARGS)
	cmake --build build
	ln -sf build/compile_commands.json .

test:
	ctest --test-dir build