#ifndef SPARSE_VECTORS_H
#define SPARSE_VECTORS_H

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "amaiss/types.h"

namespace amaiss {

enum ElementSize : uint8_t { U8 = 1, U16 = 2, U32 = 4, U64 = 8 };

struct SparseVectorsConfig {
    size_t element_size;
    size_t dimension;
};

struct SparseVectorsData {
    const idx_t* indptr_data;
    const term_t* indices_data;
    const float* values_data;
};

class SparseVectors {
    std::vector<idx_t> indptr_;
    std::vector<term_t> indices_;
    std::vector<uint8_t> values_;
    SparseVectorsConfig config_;

public:
    explicit SparseVectors(SparseVectorsConfig config);
    ~SparseVectors() = default;

    // copy constructor
    SparseVectors(const SparseVectors& other) = default;
    SparseVectors& operator=(const SparseVectors& other) = default;
    // move constructor
    SparseVectors(SparseVectors&& other) noexcept = default;
    SparseVectors& operator=(SparseVectors&& other) noexcept = default;

    void add_vectors(
        const std::vector<idx_t>& indptr, const std::vector<term_t>& indices,
        const std::vector<float>& weights);  // Get spans for the sparse

    void add_vector(const std::vector<term_t>& indices,
                    const std::vector<float>& weights);

    size_t num_vectors() const;
    size_t get_dimension() const { return config_.dimension; }
    size_t get_element_size() const { return config_.element_size; }

    std::vector<float> get_dense_vector_float(idx_t vector_idx) const;
    const idx_t* indptr_data() const { return indptr_.data(); }
    const term_t* indices_data() const { return indices_.data(); }
    const float* values_data() const {
        return reinterpret_cast<const float*>(values_.data());
    }

    SparseVectorsData get_all_data() const {
        return {.indptr_data = indptr_data(),
                .indices_data = indices_data(),
                .values_data = values_data()};
    }
};

}  // namespace amaiss

#endif  // SPARSE_VECTORS_H