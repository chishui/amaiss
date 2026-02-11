#ifndef SEISMIC_COMMON_H
#define SEISMIC_COMMON_H

#include <numeric>
#include <vector>

#include "amaiss/id_selector.h"
#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"
#include "amaiss/utils/distance_simd.h"

namespace amaiss {
namespace detail {
inline std::vector<float> calculate_summary_scores(
    const size_t element_size, const SparseVectors* summaries,
    const std::vector<uint8_t>& dense) {
    std::vector<float> summary_scores;
    if (element_size == U32) {
        summary_scores = dot_product_float_vectors_dense(
            summaries, reinterpret_cast<const float*>(dense.data()));
    } else if (element_size == U16) {
        summary_scores = dot_product_uint16_vectors_dense(
            summaries, reinterpret_cast<const uint16_t*>(dense.data()));
    } else {
        summary_scores =
            dot_product_uint8_vectors_dense(summaries, dense.data());
    }
    return summary_scores;
}

inline float compute_similarity(idx_t doc_id, const idx_t* indptr,
                                const term_t* indices, const uint8_t* values,
                                const uint8_t* dense, size_t element_size) {
    const idx_t start = indptr[doc_id];
    const size_t len = indptr[doc_id + 1] - start;
    float score = 0.0F;
    if (element_size == U32) {
        const auto* float_values =
            reinterpret_cast<const float*>(values + start * sizeof(float));
        const auto* float_dense = reinterpret_cast<const float*>(dense);
        score = dot_product_float_dense(indices + start, float_values, len,
                                        float_dense);
    } else if (element_size == U16) {
        // start is element index, need to convert to byte offset for
        // uint16_t access
        const auto* int16_values = reinterpret_cast<const uint16_t*>(
            values + start * sizeof(uint16_t));
        const auto* int16_dense = reinterpret_cast<const uint16_t*>(dense);
        score = dot_product_uint16_dense(indices + start, int16_values, len,
                                         int16_dense);
    } else {
        score = dot_product_uint8_dense(indices + start, values + start, len,
                                        dense);
    }
    return score;
}

inline std::vector<size_t> reorder_clusters(
    const std::vector<float>& summary_scores, bool first_list) {
    std::vector<size_t> cluster_order(summary_scores.size());
    std::iota(cluster_order.begin(), cluster_order.end(), 0);
    if (first_list) {
        std::ranges::sort(cluster_order, [&](size_t a, size_t b) {
            return summary_scores[a] > summary_scores[b];
        });
    }
    return cluster_order;
}

inline bool should_run_exact_match(const IDSelector* id_selector, int k,
                                   const SparseVectors* queries) {
    if (id_selector == nullptr) {
        return false;
    }
    const auto* id_selector_enumerable =
        dynamic_cast<const IDSelectorEnumerable*>(id_selector);
    if (id_selector_enumerable == nullptr) {
        return false;
    }
    return id_selector_enumerable->size() <= k;
}
}  // namespace detail
}  // namespace amaiss

#endif  // SEISMIC_COMMON_H