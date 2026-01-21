#ifndef DISTANCE_AVX512_H
#define DISTANCE_AVX512_H

#include <immintrin.h>

#include <span>
#include <vector>

#include "amaiss/types.h"
#include "amaiss/utils/dense_vector_matrix.h"

namespace amaiss {

// matrix is a dimension * num_clusters matrix
std::vector<float> dot_product_sparse_matrix(std::span<const term_t> indices,
                                             std::span<const float> weights,
                                             const DenseVectorMatrix& matrix) {
    size_t rows =
        matrix.get_rows();  // this is actually dimension of dense vector
    size_t dimension = matrix.get_dimension();  // this is number of vectors
    size_t num_clusters = dimension;

    std::vector<float> similarities(num_clusters, 0.0F);
    size_t sparse_vector_size = indices.size();
    for (size_t i = 0; i < sparse_vector_size; ++i) {
        size_t dim = indices[i];
        if (dim >= dimension) {
            continue;
        }
        float doc_value = weights[i];

        const float* centroid_values = matrix.data() + dim * rows;

        size_t centroid_idx = 0;

        const __m512 doc_vec_512 = _mm512_set1_ps(doc_value);

        for (; centroid_idx + 32 <= num_clusters; centroid_idx += 32) {
            __m512 centroid_vec_0 =
                _mm512_loadu_ps(centroid_values + centroid_idx);
            __m512 centroid_vec_1 =
                _mm512_loadu_ps(centroid_values + centroid_idx + 16);

            __m512 current_0 = _mm512_loadu_ps(&similarities[centroid_idx]);
            __m512 current_1 =
                _mm512_loadu_ps(&similarities[centroid_idx + 16]);

            current_0 = _mm512_fmadd_ps(centroid_vec_0, doc_vec_512, current_0);
            current_1 = _mm512_fmadd_ps(centroid_vec_1, doc_vec_512, current_1);

            _mm512_storeu_ps(&similarities[centroid_idx], current_0);
            _mm512_storeu_ps(&similarities[centroid_idx + 16], current_1);
        }

        for (; centroid_idx + 16 <= num_clusters; centroid_idx += 16) {
            __m512 centroid_vec =
                _mm512_loadu_ps(centroid_values + centroid_idx);
            __m512 current = _mm512_loadu_ps(&similarities[centroid_idx]);

            current = _mm512_fmadd_ps(centroid_vec, doc_vec_512, current);

            _mm512_storeu_ps(&similarities[centroid_idx], current);
        }

        const __m256 doc_vec_256 = _mm256_set1_ps(doc_value);

        for (; centroid_idx + 8 <= num_clusters; centroid_idx += 8) {
            __m256 centroid_vec =
                _mm256_loadu_ps(centroid_values + centroid_idx);
            __m256 current = _mm256_loadu_ps(&similarities[centroid_idx]);

            current = _mm256_fmadd_ps(centroid_vec, doc_vec_256, current);

            _mm256_storeu_ps(&similarities[centroid_idx], current);
        }

        for (; centroid_idx < num_clusters; ++centroid_idx) {
            similarities[centroid_idx] +=
                doc_value * centroid_values[centroid_idx];
        }
    }

    return similarities;
}

// Find the index of the maximum value in a vector using SIMD
inline size_t argmax_simd(const std::vector<float>& values) {
    if (values.empty()) {
        return 0;
    }

    size_t size = values.size();
    const float* data = values.data();

    // Handle small arrays with scalar code
    if (size < 16) {
        return argmax_scalar(values);
    }

    // SIMD processing - initialize with first 16 elements
    size_t i = 0;
    __m512 max_vec = _mm512_loadu_ps(data);
    __m512i idx_vec =
        _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i offset_vec = _mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 23,
                                          22, 21, 20, 19, 18, 17, 16);
    __m512i increment = _mm512_set1_epi32(16);

    i = 16;

    // Process 16 floats at a time
    for (; i + 16 <= size; i += 16) {
        __m512 current = _mm512_loadu_ps(data + i);
        __mmask16 mask = _mm512_cmp_ps_mask(current, max_vec, _CMP_GT_OQ);
        max_vec = _mm512_mask_blend_ps(mask, max_vec, current);
        idx_vec = _mm512_mask_blend_epi32(mask, idx_vec, offset_vec);
        offset_vec = _mm512_add_epi32(offset_vec, increment);
    }

    // Horizontal reduction to find max and its index
    alignas(64) float max_vals[16];
    alignas(64) int indices[16];
    _mm512_store_ps(max_vals, max_vec);
    _mm512_store_si512(reinterpret_cast<__m512i*>(indices), idx_vec);

    float final_max = max_vals[0];
    size_t final_idx = indices[0];
    for (int j = 1; j < 16; ++j) {
        if (max_vals[j] > final_max) {
            final_max = max_vals[j];
            final_idx = indices[j];
        }
    }

    // Handle remaining elements
    for (; i < size; ++i) {
        if (data[i] > final_max) {
            final_max = data[i];
            final_idx = i;
        }
    }

    return final_idx;
}

}  // namespace amaiss

#endif  // DISTANCE_AVX512_H