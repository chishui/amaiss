#include "amaiss/sparse_vectors.h"

#include <stdexcept>
#include <vector>

#include "amaiss/types.h"

namespace amaiss {
SparseVectors::SparseVectors(SparseVectorsConfig config) : config_(config) {}

void SparseVectors::add_vectors(const std::vector<idx_t>& indptr,
                                const std::vector<term_t>& indices,
                                const std::vector<float>& weights) {
    if (indptr.empty() || indptr.size() < 2) {
        return;  // Nothing to add
    }

    // Get the offset for the new indices
    idx_t offset = this->indptr_.empty() ? 0 : this->indptr_.back();

    // Append new indices and weights
    this->indices_.insert(this->indices_.end(), indices.begin(), indices.end());
    // Convert float vector to bytes and insert into values_
    const auto* bytes = reinterpret_cast<const uint8_t*>(weights.data());
    this->values_.insert(this->values_.end(), bytes,
                         bytes + (weights.size() * sizeof(float)));

    // Append new indptr values
    // If this is the first add, include the first element (0)
    // Otherwise skip it as it's redundant with the last element
    size_t start_idx = this->indptr_.empty() ? 0 : 1;
    for (size_t i = start_idx; i < indptr.size(); ++i) {
        this->indptr_.push_back(indptr[i] + offset);
    }
}

void SparseVectors::add_vector(const std::vector<term_t>& indices,
                               const std::vector<float>& weights) {
    if (indices.size() != weights.size()) {
        throw std::invalid_argument(
            "Indices and weights must have the same size");
    }

    // Get the current offset (where the new vector starts)
    idx_t offset = this->indptr_.empty() ? 0 : this->indptr_.back();

    // If this is the first vector, initialize indptr with 0
    if (this->indptr_.empty()) {
        this->indptr_.push_back(0);
    }

    // Append indices
    this->indices_.insert(this->indices_.end(), indices.begin(), indices.end());

    // Convert float weights to bytes and append to values_
    const auto* bytes = reinterpret_cast<const uint8_t*>(weights.data());
    this->values_.insert(this->values_.end(), bytes,
                         bytes + (weights.size() * sizeof(float)));

    // Update indptr with the new end position
    this->indptr_.push_back(offset + static_cast<idx_t>(indices.size()));
}

// Get spans for the sparse vector at the given index
SparseVectorView SparseVectors::get_vector_view(idx_t vector_idx) const {
    if (vector_idx < 0 || vector_idx > static_cast<idx_t>(indptr_.size()) - 2) {
        throw std::out_of_range("Vector index out of range");
    }

    idx_t start = indptr_[vector_idx];
    idx_t end = indptr_[vector_idx + 1];
    size_t size = end - start;

    return {.indices = std::span<const term_t>(indices_.data() + start, size),
            .values = std::span<const float>(
                reinterpret_cast<const float*>(values_.data() +
                                               (start * config_.element_size)),
                size)};
}

SparseVectorViewCoded SparseVectors::get_vector_view_coded(
    idx_t vector_idx) const {
    if (vector_idx < 0 || vector_idx > static_cast<idx_t>(indptr_.size()) - 2) {
        throw std::out_of_range("Vector index out of range");
    }

    idx_t start = indptr_[vector_idx];
    idx_t end = indptr_[vector_idx + 1];
    size_t size = end - start;
    const size_t& element_size = config_.element_size;
    return SparseVectorViewCoded{
        .indices = std::span<const term_t>(indices_.data() + start, size),
        .values = std::span<const uint8_t>(
            values_.data() + (start * element_size), size * element_size)};
}

std::vector<float> SparseVectors::get_dense_vector_float(
    idx_t vector_idx) const {
    if (vector_idx < 0 || vector_idx > static_cast<idx_t>(indptr_.size()) - 2) {
        throw std::out_of_range("Vector index out of range");
    }

    idx_t start = indptr_[vector_idx];
    idx_t end = indptr_[vector_idx + 1];
    if (config_.dimension == 0 && (end < 1 || indices_[end - 1] + 1 < 0)) {
        throw std::out_of_range("Vector index out of range");
    }
    size_t size = end - start;
    std::vector<float> dense_vector(
        config_.dimension > 0 ? config_.dimension : indices_[end - 1] + 1,
        0.0F);
    for (idx_t i = start; i < end; ++i) {
        dense_vector[indices_[i]] = *reinterpret_cast<const float*>(
            values_.data() + (i * config_.element_size));
    }
    return dense_vector;
}

size_t SparseVectors::num_vectors() const {
    if (indptr_.empty()) return 0;
    return indptr_.size() - 1;
}
}  // namespace amaiss