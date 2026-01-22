#include "amaiss/brutal_index.h"

#include <algorithm>
#include <format>
#include <iostream>
#include <memory>
#include <vector>

#include "amaiss/sparse_vectors.h"
#include "amaiss/utils/distance.h"
#include "amaiss/utils/print.h"
#include "amaiss/utils/ranker.h"

namespace amaiss {

BrutalIndex::BrutalIndex(int dim) : Index(dim) {}

void BrutalIndex::add(idx_t n, std::vector<idx_t>& indptr,
                      std::vector<term_t>& indices,
                      std::vector<float>& values) {
    if (vectors_ == nullptr) {
        vectors_ = std::unique_ptr<SparseVectors>(
            new SparseVectors({.element_size = ElementSize::U32,
                               .dimension = static_cast<size_t>(dimension_)}));
    }
    vectors_->add_vectors(indptr, indices, values);
}

auto BrutalIndex::search(idx_t n, std::vector<idx_t>& indptr,
                         std::vector<term_t>& indices,
                         std::vector<float>& values, int k)
    -> std::vector<std::vector<idx_t>> {
    if (vectors_ == nullptr || n == 0) {
        return std::vector<std::vector<idx_t>>(n);
    }

    // Create query vectors from input
    SparseVectors query_vectors({.element_size = ElementSize::U32,
                                 .dimension = static_cast<size_t>(dimension_)});
    query_vectors.add_vectors(indptr, indices, values);
    std::vector<std::vector<idx_t>> results(n);

    // For each query vector
#pragma omp parallel for
    for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
        auto dense = query_vectors.get_dense_vector_float(query_idx);
        results[query_idx] = single_query(dense, k);
    }

    return results;
}

auto BrutalIndex::single_query(const std::vector<float>& dense, int k)
    -> std::vector<idx_t> {
    DedupeTopKHolder<idx_t> holder(k);
    size_t num_docs = vectors_->num_vectors();
    if (num_docs == 0) {
        return std::vector<idx_t>();
    }
    for (size_t i = 0; i < num_docs; ++i) {
        const auto& [indices, weights] = vectors_->get_vector_view(i);
        float score = dot_product_float_dense(indices, weights, dense);
        holder.add(score, i);
    }
    auto results = holder.top_k();
    std::reverse(results.begin(), results.end());
    results.resize(k);
    return results;
}

const SparseVectors* BrutalIndex::get_vectors() const { return vectors_.get(); }

}  // namespace amaiss