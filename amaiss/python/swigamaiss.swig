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
#include "amaiss/cluster/random_kmeans.h"
%}

%init %{
import_array();
%}

%include "amaiss_typemaps.i"

%include "amaiss/types.h"
%include "amaiss/sparse_vectors.h"
%include "amaiss/invlists/inverted_lists.h"

// Ignore the protected std::vector-based methods
%ignore amaiss::Index::add(idx_t, std::vector<idx_t>&, std::vector<term_t>&, std::vector<float>&);
%ignore amaiss::Index::search(idx_t, std::vector<idx_t>&, std::vector<term_t>&, std::vector<float>&, int);

// Ignore SeismicIndex's vector-based search methods, expose only pointer-based
%ignore amaiss::SeismicIndex::search(idx_t, std::vector<idx_t>&, std::vector<term_t>&, std::vector<float>&, int);
%ignore amaiss::SeismicIndex::search(idx_t, std::vector<idx_t>&, std::vector<term_t>&, std::vector<float>&, int, int, float);

%include "amaiss/index.h"
%include "amaiss/brutal_index.h"
%include "amaiss/seismic_index.h"
%include "amaiss/cluster/random_kmeans.h"
