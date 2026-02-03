#include "amaiss/sparse_vectors.h"

#include <stdexcept>
#include <vector>

#include "amaiss/types.h"
#include "amaiss/utils/checks.h"

namespace amaiss {
SparseVectors::SparseVectors(SparseVectorsConfig config) : config_(config) {
    throw_if_not_positive(config_.dimension);
}

void SparseVectors::add_vectors(const std::vector<idx_t>& indptr,
                                const std::vector<term_t>& indices,
                                const std::vector<uint8_t>& weights) {
    add_vectors(indptr.data(), indptr.size(), indices.data(), indices.size(),
                weights.data(), weights.size());
}

void SparseVectors::add_vectors(const idx_t* indptr, size_t indptr_size,
                                const term_t* indices, size_t indices_size,
                                const uint8_t* weights, size_t weights_size) {
    if (indices_size * config_.element_size != weights_size) {
        throw std::invalid_argument(
            "Indices and weights must have the same size");
    }
    if (indptr_size < 2) {
        return;  // Nothing to add
    }
    if (this->indptr_.empty()) {
        this->indptr_.push_back(0);
    }

    // Append new indices
    this->indices_.insert(this->indices_.end(), indices,
                          indices + indices_size);

    // Append weights directly (already in uint8_t format)
    this->values_.insert(this->values_.end(), weights, weights + weights_size);
    idx_t offset = this->indptr_.back();
    for (size_t i = 1; i < indptr_size; ++i) {
        this->indptr_.push_back(indptr[i] + offset);
    }
}

void SparseVectors::add_vector(const std::vector<term_t>& indices,
                               const std::vector<uint8_t>& weights) {
    add_vector(indices.data(), indices.size(), weights.data(), weights.size());
}

void SparseVectors::add_vector(const term_t* indices, size_t indices_size,
                               const uint8_t* weights, size_t weights_size) {
    // Get the current offset (where the new vector starts)
    idx_t offset = this->indptr_.empty() ? 0 : this->indptr_.back();

    // If this is the first vector, initialize indptr with 0
    if (this->indptr_.empty()) {
        this->indptr_.push_back(0);
    }

    this->indices_.insert(this->indices_.end(), indices,
                          indices + indices_size);
    this->values_.insert(this->values_.end(), weights, weights + weights_size);
    this->indptr_.push_back(offset + static_cast<idx_t>(indices_size));
}

std::vector<float> SparseVectors::get_dense_vector_float(
    idx_t vector_idx) const {
    if (vector_idx < 0 || vector_idx > static_cast<idx_t>(indptr_.size()) - 2) {
        throw std::out_of_range("Vector index out of range");
    }

    idx_t start = indptr_[vector_idx];
    idx_t end = indptr_[vector_idx + 1];
    std::vector<float> dense_vector(
        config_.dimension > 0 ? config_.dimension : indices_[end - 1] + 1,
        0.0F);
    for (idx_t i = start; i < end; ++i) {
        const uint8_t* value_ptr = values_.data() + (i * config_.element_size);
        if (config_.element_size == U32) {
            dense_vector[indices_[i]] =
                *reinterpret_cast<const float*>(value_ptr);
        } else if (config_.element_size == U16) {
            dense_vector[indices_[i]] = static_cast<float>(
                *reinterpret_cast<const uint16_t*>(value_ptr));
        } else {
            dense_vector[indices_[i]] = static_cast<float>(*value_ptr);
        }
    }
    return dense_vector;
}

std::vector<uint8_t> SparseVectors::get_dense_vector(idx_t vector_idx) const {
    if (vector_idx < 0 || vector_idx > static_cast<idx_t>(indptr_.size()) - 2) {
        throw std::out_of_range("Vector index out of range");
    }
    idx_t start = indptr_[vector_idx];
    idx_t end = indptr_[vector_idx + 1];
    size_t size = end - start;
    std::vector<uint8_t> dense_vector(config_.dimension * config_.element_size,
                                      0.0F);
    for (idx_t i = start; i < end; ++i) {
        for (idx_t j = 0; j < config_.element_size; ++j) {
            dense_vector[indices_[i] * config_.element_size + j] =
                values_[i * config_.element_size + j];
        }
    }
    return dense_vector;
}

size_t SparseVectors::num_vectors() const {
    if (indptr_.empty()) return 0;
    return indptr_.size() - 1;
}
}  // namespace amaiss