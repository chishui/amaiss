#ifndef DISTANCE_AVX2_H
#define DISTANCE_AVX2_H
#include <immintrin.h>

#include <span>

#include "amaiss/types.h"

namespace amaiss {
inline float dot_product_float_dense(std::span<const term_t> indices,
                                     std::span<const float> weights,
                                     const std::vector<float>& dense) {
    // Float version using AVX2
    size_t size = indices.size();
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        float gathered[8];
        for (int j = 0; j < 8; ++j) gathered[j] = dense[indices[i + j]];
        __m256 vals = _mm256_loadu_ps(&weights[i]);
        __m256 dense_vals = _mm256_loadu_ps(gathered);
        sum = _mm256_fmadd_ps(vals, dense_vals, sum);
    }
    float temp[8];
    _mm256_storeu_ps(temp, sum);
    float result = 0.0f;
    for (int j = 0; j < 8; ++j) result += temp[j];
    for (; i < size; ++i) result += weights[i] * dense[indices[i]];
    return result;
}

}  // namespace amaiss
#endif  // DISTANCE_AVX2_H