#include "amaiss/index.h"

#include <algorithm>

#include "amaiss/types.h"
#include "amaiss/utils/checks.h"

namespace amaiss {

Index::Index(int dim) : dimension_(dim) {}

void Index::build() { throw_not_implemented(); }

void Index::search(idx_t n, const idx_t* indptr, const term_t* indices,
                   const float* values, int k, idx_t* labels,
                   const SearchParameters* search_parameters) {
    throw_if_not_positive(n);
    throw_if_not_positive(k);
    throw_if_any_null(indptr, indices, values, labels);

    auto results = search(n, indptr, indices, values, k, search_parameters);

    idx_t* dest = labels;
    for (const auto& result : results) {
        dest = std::ranges::copy(result, dest).out;
    }
}

auto Index::search(idx_t n, const idx_t* indptr, const term_t* indices,
                   const float* values, int k,
                   const SearchParameters* search_parameters)
    -> std::vector<std::vector<idx_t>> {
    throw_not_implemented("search not implementted in Index");
}

void Index::add_with_ids(idx_t n, const idx_t* indptr, const term_t* indices,
                         const float* values, const idx_t* ids) {
    throw_not_implemented("add_with_ids not implemented in Index");
}

}  // namespace amaiss