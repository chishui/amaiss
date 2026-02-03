#include "amaiss/cluster/inverted_list_clusters.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

namespace amaiss {
namespace {

/**
 * @brief Generage summary sparse vector for posting lists
 *
 * @param vectors inverted index
 * @param group_of_doc_ids a list of posting list
 * @param alpha prune ratio
 * @return SparseVectors
 */
template <class T>
SparseVectors summarize_(const SparseVectors* vectors,
                         const std::vector<idx_t>& group_of_doc_ids,
                         const std::vector<idx_t>& offsets, float alpha) {
    SparseVectors summarized_vectors(
        {.element_size = vectors->get_element_size(),
         .dimension = vectors->get_dimension()});
    if (offsets.size() <= 1) {
        return summarized_vectors;
    }
    const auto element_size = vectors->get_element_size();
    const auto& indptr_data = vectors->indptr_data();
    const auto& indices_data = vectors->indices_data();
    const auto& values_data = vectors->values_data();
    for (size_t i = 0; i < offsets.size() - 1; ++i) {
        size_t n_docs = offsets[i + 1] - offsets[i];
        std::unordered_map<term_t, T> summary_map;
        float sum = 0.0F;
        auto doc_ids = std::span<const idx_t>(
            group_of_doc_ids.data() + offsets[i], n_docs);
        for (const auto& doc_id : doc_ids) {
            int start = indptr_data[doc_id];
            int end = indptr_data[doc_id + 1];
            for (size_t j = start; j < end; ++j) {
                const auto old = summary_map[indices_data[j]];
                auto& value = summary_map[indices_data[j]];
                // j is element index, need byte offset for T access
                value = std::max(value, *reinterpret_cast<const T*>(
                                            values_data + j * sizeof(T)));
                sum += value - old;
            }
        }

        // Convert summary_map to vector of pairs
        std::vector<std::pair<term_t, T>> summary_vec(summary_map.begin(),
                                                      summary_map.end());

        // Sort by value in descending order
        std::ranges::sort(summary_vec, [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        float addup = 0.0F;
        for (int j = 0; j < summary_vec.size(); ++j) {
            addup += summary_vec[j].second;
            if (addup / sum >= alpha) {
                summary_vec.erase(summary_vec.begin() + j + 1,
                                  summary_vec.end());
                break;
            }
        }

        // Sort by term_t order
        std::ranges::sort(summary_vec, [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        // Break into separate terms and values vectors
        std::vector<term_t> terms;
        std::vector<T> values;
        terms.reserve(summary_vec.size());
        values.reserve(summary_vec.size());
        for (const auto& [term, value] : summary_vec) {
            terms.push_back(term);
            values.push_back(value);
        }

        summarized_vectors.add_vector(
            terms.data(), terms.size(),
            reinterpret_cast<const uint8_t*>(values.data()),
            values.size() * sizeof(T));
    }
    return summarized_vectors;
}

}  // namespace

InvertedListClusters::InvertedListClusters(
    const std::vector<std::vector<idx_t>>& docs) {
    if (docs.empty()) return;
    offsets_.reserve(docs.size() + 1);
    offsets_.push_back(0);
    for (const auto& doc_ids : docs) {
        docs_.insert(docs_.end(), doc_ids.begin(), doc_ids.end());
        offsets_.push_back(docs_.size());
    }
}

InvertedListClusters::InvertedListClusters(const InvertedListClusters& other) {
    docs_ = other.docs_;
    offsets_ = other.offsets_;
    if (other.summaries_ != nullptr) {
        summaries_ = std::make_unique<SparseVectors>(*other.summaries_);
    } else {
        summaries_.reset();
    }
}
InvertedListClusters& InvertedListClusters::operator=(
    const InvertedListClusters& other) {
    if (this != &other) {
        docs_ = other.docs_;
        offsets_ = other.offsets_;
        if (other.summaries_ != nullptr) {
            summaries_ = std::make_unique<SparseVectors>(*other.summaries_);
        } else {
            summaries_.reset();
        }
    }
    return *this;
}

auto InvertedListClusters::get_docs(idx_t idx) const -> std::span<const idx_t> {
    return {docs_.data() + offsets_[idx],
            static_cast<size_t>(offsets_[idx + 1] - offsets_[idx])};
}

void InvertedListClusters::summarize(const SparseVectors* vectors,
                                     float alpha) {
    if (summaries_ != nullptr) {
        summaries_.reset();
    }
    const auto element_size = vectors->get_element_size();
    if (element_size == U32) {
        summaries_ = std::make_unique<SparseVectors>(
            std::move(summarize_<float>(vectors, docs_, offsets_, alpha)));
    } else if (element_size == U16) {
        summaries_ = std::make_unique<SparseVectors>(
            std::move(summarize_<uint16_t>(vectors, docs_, offsets_, alpha)));
    } else {
        summaries_ = std::make_unique<SparseVectors>(
            std::move(summarize_<uint8_t>(vectors, docs_, offsets_, alpha)));
    }
}

}  // namespace amaiss