#ifndef INVERTED_LIST_CLUSTERS_H
#define INVERTED_LIST_CLUSTERS_H
#include <memory>
#include <vector>

#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"
#include "amaiss/utils/dense_vector_matrix.h"

namespace amaiss {

class InvertedListClusters {
public:
    InvertedListClusters() = default;
    InvertedListClusters(const std::vector<std::vector<idx_t>>& docs);
    // copy constructor
    InvertedListClusters(const InvertedListClusters& other);
    InvertedListClusters& operator=(const InvertedListClusters& other);
    // move constructor
    InvertedListClusters(InvertedListClusters&& other) noexcept = default;
    InvertedListClusters& operator=(InvertedListClusters&& other) noexcept =
        default;
    virtual ~InvertedListClusters() = default;

    auto get_docs(idx_t idx) const -> std::span<const idx_t>;

    auto get_summary(idx_t i) const -> const SparseVectorView {
        if (summaries_ == nullptr) return {};
        return summaries_->get_vector_view(i);
    }
    auto summaries() const -> const SparseVectors& { return *summaries_; }

    void summarize(const SparseVectors* vectors, float alpha);

    size_t cluster_size() const {
        return summaries_ == nullptr ? 0 : summaries_->num_vectors();
    }

private:
    std::vector<idx_t> docs_;
    std::vector<idx_t> offsets_;
    std::unique_ptr<SparseVectors> summaries_;
};

}  // namespace amaiss

#endif  // INVERTED_LIST_CLUSTERS_H