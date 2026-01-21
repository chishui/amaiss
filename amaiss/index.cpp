#include "amaiss/index.h"

#include <algorithm>

#include "amaiss/types.h"
#include "amaiss/utils/checks.h"

namespace amaiss {

Index::Index(int dim) : dimension_(dim) {}

void Index::add(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values) {
    throw_if_not_positive(n);
    throw_if_any_null(indptr, indices, values);

    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];  // Total non-zeros

    // Direct construction - single allocation + copy
    std::vector<idx_t> indptr_vec(indptr, indptr + indptr_size);
    std::vector<term_t> indices_vec(indices, indices + nnz);
    std::vector<float> values_vec(values, values + nnz);

    add(n, indptr_vec, indices_vec, values_vec);
}

void Index::search(idx_t n, const idx_t* indptr, const term_t* indices,
                   const float* values, int k, idx_t* labels) {
    throw_if_not_positive(n);
    throw_if_not_positive(k);
    throw_if_any_null(indptr, indices, values, labels);

    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];  // Total non-zeros

    // Direct construction - single allocation + copy
    std::vector<idx_t> indptr_vec(indptr, indptr + indptr_size);
    std::vector<term_t> indices_vec(indices, indices + nnz);
    std::vector<float> values_vec(values, values + nnz);

    auto results = search(n, indptr_vec, indices_vec, values_vec, k);

    idx_t* dest = labels;
    for (const auto& result : results) {
        dest = std::ranges::copy(result, dest).out;
    }
}

}  // namespace amaiss