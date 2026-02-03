#ifndef DISTANCE_H
#define DISTANCE_H
#include <cstddef>
#include <span>
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
                                     const float* dense) {
    float result = 0.0F;
    for (size_t i = 0; i < len; ++i) {
        result += weights[i] * dense[indices[i]];
    }
    return result;
}

inline float dot_product_uint8_dense(const term_t* indices,
                                     const uint8_t* weights, size_t len,
                                     const uint8_t* dense) {
    int result = 0;
    for (size_t i = 0; i < len; ++i) {
        result += weights[i] * dense[indices[i]];
    }
    return static_cast<float>(result);
}

inline float dot_product_uint16_dense(const term_t* indices,
                                      const uint16_t* weights, size_t len,
                                      const uint16_t* dense) {
    int64_t result = 0;
    for (size_t i = 0; i < len; ++i) {
        result += static_cast<int64_t>(weights[i]) * dense[indices[i]];
    }
    return static_cast<float>(result);
}

inline auto dot_product_float_dense(const SparseVectors* vectors,
                                    std::span<const float> dense)
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

template <class T>
inline auto dot_product_dense(const SparseVectors* vectors, const T* dense)
    -> std::vector<float> {
    size_t n_vectors = vectors->num_vectors();
    std::vector<float> results(n_vectors, 0.0F);

    const auto* indptr = vectors->indptr_data();
    const auto* indices = vectors->indices_data();
    // values_data() returns uint8_t*, indptr stores element indices
    // so we need to access values at byte offset = element_index * sizeof(T)
    const auto* values = vectors->values_data();

    for (size_t i = 0; i < n_vectors; ++i) {
        const idx_t start = indptr[i];
        const idx_t end = indptr[i + 1];
        const size_t len = end - start;
        const term_t* idx_ptr = indices + start;
        // Cast to T* at the correct byte offset
        const T* val_ptr =
            reinterpret_cast<const T*>(values + start * sizeof(T));

        for (size_t j = 0; j < len; ++j) {
            auto index = idx_ptr[j];
            if (dense[index] == 0) {
                continue;
            }
            results[i] += static_cast<float>(val_ptr[j]) * dense[index];
        }
    }
    return results;
}

}  // namespace amaiss

#endif  // DISTANCE_H