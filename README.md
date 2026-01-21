# Amaiss

## Developer Guide

### Prerequisites

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y g++ cmake libomp-dev python3.12-dev python3-pip swig
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

```bash
make test
```

### Python Usage

After building, you can use the Python bindings:

```bash
cd amaiss/python
pip install -e .
```

Then run the example:

```bash
python demos/amaiss_example.py
```

### Clean Build

To remove all build artifacts and start fresh:

```bash
make clean
```