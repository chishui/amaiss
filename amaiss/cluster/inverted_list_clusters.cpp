#include "amaiss/cluster/inverted_list_clusters.h"

#include <algorithm>
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
SparseVectors summarize_(const SparseVectors* vectors,
                         const std::vector<idx_t>& group_of_doc_ids,
                         const std::vector<idx_t>& offsets, float alpha) {
    SparseVectors summarized_vectors(
        {.element_size = vectors->get_element_size(),
         .dimension = vectors->get_dimension()});
    if (offsets.size() <= 1) {
        return summarized_vectors;
    }
    for (size_t i = 0; i < offsets.size() - 1; ++i) {
        size_t n_docs = offsets[i + 1] - offsets[i];
        std::unordered_map<term_t, float> summary_map;
        float sum = 0.0F;
        auto doc_ids = std::span<const idx_t>(
            group_of_doc_ids.data() + offsets[i], n_docs);
        for (const auto& doc_id : doc_ids) {
            const auto& [indices, values] = vectors->get_vector_view(doc_id);
            for (size_t j = 0; j < indices.size(); ++j) {
                auto old = summary_map[indices[j]];
                auto& value = summary_map[indices[j]];
                value = std::max(value, values[j]);
                sum += value - old;
            }
        }

        // Convert summary_map to vector of pairs
        std::vector<std::pair<term_t, float>> summary_vec(summary_map.begin(),
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
        std::vector<float> values;
        terms.reserve(summary_vec.size());
        values.reserve(summary_vec.size());
        for (const auto& [term, value] : summary_vec) {
            terms.push_back(term);
            values.push_back(value);
        }

        summarized_vectors.add_vector(terms, values);
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
    // Need idx + 1 to be valid, so check offsets_.size() <= idx + 1
    if ((idx < 0) || (offsets_.size() <= static_cast<size_t>(idx) + 1) ||
        (offsets_[idx] == offsets_[idx + 1])) {
        return {};
    }
    return {docs_.data() + offsets_[idx],
            static_cast<size_t>(offsets_[idx + 1] - offsets_[idx])};
}

void InvertedListClusters::summarize(const SparseVectors* vectors,
                                     float alpha) {
    if (summaries_ != nullptr) {
        summaries_.reset();
    }
    auto summarized_vectors = summarize_(vectors, docs_, offsets_, alpha);
    summaries_ = std::make_unique<SparseVectors>(std::move(summarized_vectors));
}

}  // namespace amaiss