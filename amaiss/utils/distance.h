#ifndef DISTANCE_H
#define DISTANCE_H
#include <iostream>
#include <span>
#include <vector>

#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

namespace amaiss {
/**
 * @brief sparse X dense
 *
 * @param indices1
 * @param weights1
 * @param dense dense vector is expected to be of size max_index+1
 * @return float
 */
inline float dot_product_float_dense(std::span<const term_t> indices1,
                                     std::span<const float> weights1,
                                     const std::vector<float>& dense) {
    float result = 0.0F;
    size_t size = indices1.size();
    for (size_t i = 0; i < size; ++i) {
        result += weights1[i] * dense[indices1[i]];
    }

    return result;
}

inline auto dot_product_float_dense(const SparseVectors* vectors,
                                    const std::vector<float>& dense)
    -> std::vector<float> {
    size_t n_vectors = vectors->num_vectors();
    std::vector<float> results(n_vectors, 0.0F);
    for (size_t i = 0; i < n_vectors; ++i) {
        const auto& [indices, weights] = vectors->get_vector_view(i);
        for (int j = 0; j < indices.size(); ++j) {
            auto index = indices[j];
            if (index >= dense.size()) {
                break;
            }
            if (dense[index] == 0) {
                continue;
            }
            results[i] += weights[j] * dense[index];
        }
    }
    return results;
}

}  // namespace amaiss

#endif  // DISTANCE_H