%module swigamaiss

%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "amaiss/types.h"
#include "amaiss/sparse_vectors.h"
#include "amaiss/invlists/inverted_lists.h"
#include "amaiss/index.h"
#include "amaiss/brutal_index.h"
#include "amaiss/seismic_index.h"
#include "amaiss/seismic_scalar_quantized_index.h"
#include "amaiss/utils/scalar_quantizer.h"
#include "amaiss/cluster/random_kmeans.h"
#include "amaiss/index_factory.h"
%}

%init %{
import_array();
%}

%include "amaiss_typemaps.i"

%include "amaiss/types.h"
%include "amaiss/sparse_vectors.h"
%include "amaiss/invlists/inverted_lists.h"
%include "amaiss/utils/scalar_quantizer.h"

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
// (labels has numinputs=0, so both would appear as 5-arg functions to Python)
%extend amaiss::Index {
    // Version without SearchParameters
    void search_c(idx_t n, const idx_t* indptr, const term_t* indices,
                  const float* values, int k, idx_t* labels) {
        $self->search(n, indptr, indices, values, k, labels, nullptr);
    }
    
    // Version with SearchParameters - different name to avoid overload ambiguity
    void search_with_params(idx_t n, const idx_t* indptr, const term_t* indices,
                            const float* values, int k, idx_t* labels,
                            const amaiss::SearchParameters* search_parameters) {
        $self->search(n, indptr, indices, values, k, labels, search_parameters);
    }
}

%include "amaiss/index.h"
%include "amaiss/brutal_index.h"
%include "amaiss/seismic_index.h"
%include "amaiss/seismic_scalar_quantized_index.h"
%include "amaiss/cluster/random_kmeans.h"

// index_factory returns a new object - Python takes ownership
%newobject amaiss::index_factory;
%include "amaiss/index_factory.h"
