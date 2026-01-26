#ifndef DISTANCE_H
#define DISTANCE_H
#include <cstddef>
#include <vector>

#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

namespace amaiss {

/**
 * @brief sparse X dense (raw pointer version for hot paths)
 *
 * @param indices pointer to sparse indices
 * @param weights pointer to sparse weights
 * @param len number of elements
 * @param dense dense vector is expected to be of size max_index+1
 * @return float
 */
inline float dot_product_float_dense(const term_t* indices,
                                     const float* weights, size_t len,
                                     const std::vector<float>& dense) {
    float result = 0.0F;
    for (size_t i = 0; i < len; ++i) {
        result += weights[i] * dense[indices[i]];
    }
    return result;
}

inline auto dot_product_float_dense(const SparseVectors* vectors,
                                    const std::vector<float>& dense)
    -> std::vector<float> {
    size_t n_vectors = vectors->num_vectors();
    std::vector<float> results(n_vectors, 0.0F);

    const auto& [indptr, indices, values] = vectors->get_all_data();

    for (size_t i = 0; i < n_vectors; ++i) {
        const idx_t start = indptr[i];
        const idx_t end = indptr[i + 1];
        const size_t len = end - start;
        const term_t* idx_ptr = indices + start;
        const float* val_ptr = values + start;

        for (size_t j = 0; j < len; ++j) {
            auto index = idx_ptr[j];
            if (dense[index] == 0) {
                continue;
            }
            results[i] += val_ptr[j] * dense[index];
        }
    }
    return results;
}

}  // namespace amaiss

#endif  // DISTANCE_H