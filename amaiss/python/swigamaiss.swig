%module swigamaiss

%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "amaiss/types.h"
#include "amaiss/io/io.h"
#include "amaiss/id_selector.h"
#include "amaiss/sparse_vectors.h"
#include "amaiss/invlists/inverted_lists.h"
#include "amaiss/index.h"
#include "amaiss/brutal_index.h"
#include "amaiss/seismic_common.h"
#include "amaiss/seismic_index.h"
#include "amaiss/seismic_scalar_quantized_index.h"
#include "amaiss/id_map_index.h"
#include "amaiss/utils/scalar_quantizer.h"
#include "amaiss/cluster/random_kmeans.h"
#include "amaiss/index_factory.h"
#include "amaiss/io/index_io.h"
%}

%init %{
import_array();
%}

%include "amaiss_typemaps.i"
%include "factory.i"
%include "amaiss/io/io.h"
%include "amaiss/types.h"
%include "amaiss/sparse_vectors.h"
%include "amaiss/invlists/inverted_lists.h"
%include "amaiss/utils/scalar_quantizer.h"

// IDSelector support - expose the class hierarchy for Python
// Multi-argument typemap: Python passes a single int32 array, n is inferred
%typemap(in) (size_t n, const amaiss::idx_t* indices) (Py_buffer view) {
    if (PyObject_GetBuffer($input, &view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) == -1) {
        SWIG_fail;
    }
    if (strcmp(view.format, "i") != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError, "Expected int32 array for indices");
        SWIG_fail;
    }
    $1 = (size_t)(view.shape[0]);
    $2 = (amaiss::idx_t*)view.buf;
}
%typemap(freearg) (size_t n, const amaiss::idx_t* indices) {
    PyBuffer_Release(&view$argnum);
}
%include "amaiss/id_selector.h"

// IO classes needed for index serialization (must come before index.h)
// Ignore internal IO classes - not needed in Python API
%ignore amaiss::SeismicIndex::write_index;
%ignore amaiss::SeismicIndex::read_index;
%ignore amaiss::SeismicScalarQuantizedIndex::write_index;
%ignore amaiss::SeismicScalarQuantizedIndex::read_index;
%ignore amaiss::IDMapIndex::write_index;
%ignore amaiss::IDMapIndex::read_index;
%ignore amaiss::SparseVectors::serialize;
%ignore amaiss::SparseVectors::deserialize;
%ignore amaiss::InvertedListClusters::serialize;
%ignore amaiss::InvertedListClusters::deserialize;
%ignore amaiss::detail; 

// Ignore the protected std::vector-based methods
%ignore amaiss::Index::add(idx_t, std::vector<idx_t>&, std::vector<term_t>&, std::vector<float>&);
%ignore amaiss::Index::search(idx_t, std::vector<idx_t>&, std::vector<term_t>&, std::vector<float>&, int);

// Ignore SeismicIndex's vector-based search methods, expose only pointer-based
%ignore amaiss::SeismicIndex::search(idx_t, std::vector<idx_t>&, std::vector<term_t>&, std::vector<float>&, int);
%ignore amaiss::SeismicIndex::search(idx_t, std::vector<idx_t>&, std::vector<term_t>&, std::vector<float>&, int, int, float);

// Ignore SeismicScalarQuantizedIndex's vector-based search methods
%ignore amaiss::SeismicScalarQuantizedIndex::search(idx_t, std::vector<idx_t>&, std::vector<term_t>&, std::vector<float>&, int);

// Ignore the original search methods - we'll provide our own via %extend
%ignore amaiss::Index::search;

// Extend Index to provide search methods
// Use separate names because both overloads have the same number of visible Python args
// (labels and distances have numinputs=0, so both would appear as 5-arg functions to Python)
%extend amaiss::Index {
    // Version without SearchParameters
    void search_c(idx_t n, const idx_t* indptr, const term_t* indices,
                  const float* values, int k, float* distances, idx_t* labels) {
        $self->search(n, indptr, indices, values, k, distances, labels, nullptr);
    }
    
    // Version with SearchParameters - different name to avoid overload ambiguity
    void search_with_params(idx_t n, const idx_t* indptr, const term_t* indices,
                            const float* values, int k, float* distances,
                            idx_t* labels,
                            amaiss::SearchParameters* search_parameters) {
        $self->search(n, indptr, indices, values, k, distances, labels, search_parameters);
    }
}

%include "amaiss/index.h"
%include "amaiss/brutal_index.h"
%include "amaiss/seismic_common.h"
%include "amaiss/seismic_index.h"
%include "amaiss/seismic_scalar_quantized_index.h"
%include "amaiss/id_map_index.h"
%include "amaiss/cluster/random_kmeans.h"

// index_factory returns a new object - Python takes ownership
// Use %factory to enable proper downcasting based on runtime type
%newobject amaiss::index_factory;
%factory(amaiss::Index* amaiss::index_factory, amaiss::BrutalIndex, amaiss::SeismicIndex, amaiss::SeismicScalarQuantizedIndex, amaiss::IDMapIndex);
%include "amaiss/index_factory.h"

// read_index returns a new object - Python takes ownership
%newobject amaiss::read_index;
%factory(amaiss::Index* amaiss::read_index, amaiss::BrutalIndex, amaiss::SeismicIndex, amaiss::SeismicScalarQuantizedIndex, amaiss::IDMapIndex);
%include "amaiss/io/index_io.h"
