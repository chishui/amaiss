"""
Setup script for amaiss Python bindings.

This setup.py works with CMake-built extensions.
Build steps:
  cmake -B build -DAMAISS_ENABLE_PYTHON=ON && cmake --build build -j
  cd build/amaiss/python && pip install .
"""

import os
import platform
import shutil
import sys
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


def find_extension():
    """Find the built extension module."""
    ext = ".pyd" if platform.system() == "Windows" else ".so"
    lib = f"_swigamaiss{ext}"

    if os.path.exists(lib):
        return lib, ext
    if os.path.exists("_swigamaiss.dylib"):
        return "_swigamaiss.dylib", ".dylib"

    print("Error: Extension module not found! Build with CMake first:")
    print("  cmake -B build -DAMAISS_ENABLE_PYTHON=ON")
    print("  cmake --build build -j")
    print("  cd build/amaiss/python && pip install .")
    sys.exit(1)


def prepare_package():
    """Prepare the amaiss package directory."""
    swigamaiss_lib, ext = find_extension()

    shutil.rmtree("amaiss", ignore_errors=True)
    os.mkdir("amaiss")

    shutil.copyfile("__init__.py", "amaiss/__init__.py")
    shutil.copyfile("swigamaiss.py", "amaiss/swigamaiss.py")

    # Python expects .so on Unix-like systems
    target_ext = ".so" if platform.system() != "Windows" else ext
    shutil.copyfile(swigamaiss_lib, f"amaiss/_swigamaiss{target_ext}")


prepare_package()
setup(distclass=BinaryDistribution)
