#ifndef SPARSE_VECTORS_H
#define SPARSE_VECTORS_H

#include <cstdint>
#include <span>
#include <vector>

#include "amaiss/types.h"

namespace amaiss {

enum ElementSize : uint8_t { U8 = 1, U16 = 2, U32 = 4, U64 = 8 };

// View of a sparse vector with spans for indices and weights
struct SparseVectorView {
    std::span<const term_t> indices;
    std::span<const float> values;
};

struct SparseVectorViewCoded {
    std::span<const term_t> indices;
    std::span<const uint8_t> values;
};

struct SparseVectorsConfig {
    size_t element_size;
    size_t dimension;
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
    SparseVectorView get_vector_view(idx_t vector_idx) const;
    SparseVectorViewCoded get_vector_view_coded(idx_t vector_idx) const;

    std::vector<float> get_dense_vector_float(idx_t vector_idx) const;
    size_t num_vectors() const;
    size_t get_dimension() const { return config_.dimension; }
    size_t get_element_size() const { return config_.element_size; }
};

}  // namespace amaiss

#endif  // SPARSE_VECTORS_H