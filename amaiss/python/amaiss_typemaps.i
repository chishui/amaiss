// Custom typemaps for converting Python buffer protocol objects to C pointers

// Typemap for scalar idx_t parameters (like 'n' in add/search methods)
%typemap(in) amaiss::idx_t {
    if (PyLong_Check($input)) {
        $1 = (amaiss::idx_t)PyLong_AsLong($input);
        if (PyErr_Occurred()) {
            SWIG_fail;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Expected an integer for idx_t parameter");
        SWIG_fail;
    }
}

%typemap(in) (const amaiss::idx_t* indptr) (Py_buffer view) {
    if (PyObject_GetBuffer($input, &view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) == -1) {
        SWIG_fail;
    }
    if (strcmp(view.format, "i") != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError, "Expected int32 array for indptr");
        SWIG_fail;
    }
    $1 = (amaiss::idx_t*)view.buf;
}

%typemap(freearg) (const amaiss::idx_t* indptr) {
    PyBuffer_Release(&view$argnum);
}

%typemap(in) (const amaiss::term_t* indices) (Py_buffer view) {
    if (PyObject_GetBuffer($input, &view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) == -1) {
        SWIG_fail;
    }
    if (strcmp(view.format, "H") != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError, "Expected uint16 array for indices");
        SWIG_fail;
    }
    $1 = (amaiss::term_t*)view.buf;
}

%typemap(freearg) (const amaiss::term_t* indices) {
    PyBuffer_Release(&view$argnum);
}

%typemap(in) (const float* values) (Py_buffer view) {
    if (PyObject_GetBuffer($input, &view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) == -1) {
        SWIG_fail;
    }
    if (strcmp(view.format, "f") != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError, "Expected float32 array for values");
        SWIG_fail;
    }
    $1 = (float*)view.buf;
}

%typemap(freearg) (const float* values) {
    PyBuffer_Release(&view$argnum);
}

// Multi-argument typemap for the add method signature to ensure proper validation
%typemap(check) (amaiss::idx_t n, const amaiss::idx_t* indptr, const amaiss::term_t* indices, const float* values) {
    // Validation is handled in C++ code, this just ensures the signature is recognized
}

// Multi-argument typemap for the entire search signature
// Store n and k in local variables that can be accessed in argout
%typemap(in, numinputs=0) amaiss::idx_t* labels (amaiss::idx_t* temp, amaiss::idx_t n_store, int k_store) {
    // Will be allocated in the check typemap
    temp = nullptr;
    $1 = temp;
}

// Use a multi-argument typemap to capture n, k, and labels together
%typemap(check) (amaiss::idx_t n, const amaiss::idx_t* indptr, const amaiss::term_t* indices, const float* values, int k, amaiss::idx_t* labels) {
    // Store n and k in the local variables from the labels typemap
    n_store7 = $1;  // n (note: labels is arg 7, so locals are numbered 7)
    k_store7 = $5;  // k
    // Allocate labels array: n * k using malloc so NumPy can free it
    $6 = (amaiss::idx_t*)malloc($1 * $5 * sizeof(amaiss::idx_t));
    if (!$6) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for labels");
        SWIG_fail;
    }
}

// Convert the labels output array to a 2D NumPy array (n x k) and return it
%typemap(argout) amaiss::idx_t* labels {
    // Create a 2D array with shape (n, k)
    npy_intp dims[2];
    dims[0] = n_store$argnum;  // n (number of queries)
    dims[1] = k_store$argnum;  // k (number of neighbors per query)
    
    PyObject* array = PyArray_SimpleNewFromData(2, dims, NPY_INT32, (void*)$1);
    if (!array) {
        free($1);
        SWIG_fail;
    }
    
    // Make NumPy own the data so it gets freed when the array is deleted
    PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);
    
    // Append to result tuple
    %append_output(array);
}

// Prevent double-free: NumPy now owns the memory
%typemap(freearg) amaiss::idx_t* labels {
    // Do nothing - NumPy owns the memory now
}

// Typemap for int k parameter in search method
%typemap(in) int k {
    if (PyLong_Check($input)) {
        $1 = (int)PyLong_AsLong($input);
        if (PyErr_Occurred()) {
            SWIG_fail;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Expected an integer for k parameter");
        SWIG_fail;
    }
}

// Typemap for int cut parameter in SeismicIndex::search
%typemap(in) int cut {
    if (PyLong_Check($input)) {
        $1 = (int)PyLong_AsLong($input);
        if (PyErr_Occurred()) {
            SWIG_fail;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Expected an integer for cut parameter");
        SWIG_fail;
    }
}

// Typemap for float heap_factor parameter in SeismicIndex::search
%typemap(in) float heap_factor {
    if (PyFloat_Check($input) || PyLong_Check($input)) {
        $1 = (float)PyFloat_AsDouble($input);
        if (PyErr_Occurred()) {
            SWIG_fail;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Expected a float for heap_factor parameter");
        SWIG_fail;
    }
}

// Multi-argument typemap for SeismicIndex::search with cut and heap_factor (8 args, labels is arg 9 including self)
%typemap(check) (amaiss::idx_t n, const amaiss::idx_t* indptr, const amaiss::term_t* indices, const float* values, int k, int cut, float heap_factor, amaiss::idx_t* labels) {
    n_store9 = $1;  // n
    k_store9 = $5;  // k
    $8 = (amaiss::idx_t*)malloc($1 * $5 * sizeof(amaiss::idx_t));
    if (!$8) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for labels");
        SWIG_fail;
    }
}
