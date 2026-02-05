# Amaiss

## Developer Guide

### Prerequisites

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y g++ cmake libomp-dev python3.12-dev python3-pip swig libabsl-dev
```

> Note: Replace `python3.12-dev` with your Python version (e.g., `python3.11-dev` for Python 3.11).

### Clone the Repository

```bash
git clone <repository-url>
cd amaiss
```

### Build

```bash
make build
```

This will:
- Create a `build` directory
- Configure the project with CMake
- Compile the C++ library and Python bindings

### Run Tests

First, build with tests enabled:

```bash
cmake -S . -B build -DAMAISS_ENABLE_TESTS=ON
cmake --build build
```

Then run the tests:

```bash
make test
```

To run specific test suites:

```bash
./build/tests/amaiss_test --gtest_filter="SparseVectors*"
./build/tests/amaiss_test --gtest_filter="IndexFactory*"
```

### Python Environment Setup

#### Using venv (Python's built-in virtual environment)

```bash
# On Ubuntu, install the venv package first
sudo apt install python3.12-venv  # Replace with your Python version

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install the package
cd amaiss/python
pip install -e .
```

#### Using Conda

```bash
# Create a new conda environment
conda create -n amaiss python=3.12

# Activate the environment
conda activate amaiss

# Install the package
cd amaiss/python
pip install -e .
```

### Python Usage

After setting up your environment and building, you can use the Python bindings:

```bash
python demos/amaiss_example.py
```

### Clean Build

To remove all build artifacts and start fresh:

```bash
make clean
```