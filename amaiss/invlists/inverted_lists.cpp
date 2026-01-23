#include "amaiss/invlists/inverted_lists.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "amaiss/types.h"

namespace amaiss {

namespace {
// RAII lock guard for spinlock
class LockGuard {
public:
    explicit LockGuard(std::atomic<uint8_t>& lock) : lock_(lock) {
        while (lock_.exchange(1, std::memory_order_acquire) != 0) {
            // Busy-wait
        }
    }
    ~LockGuard() { lock_.store(0, std::memory_order_release); }
    LockGuard(const LockGuard&) = delete;
    LockGuard& operator=(const LockGuard&) = delete;

private:
    std::atomic<uint8_t>& lock_;
};
}  // namespace

InvertedList::InvertedList(size_t element_size) : element_size_(element_size) {}

void InvertedList::add_entries(size_t n_entry, const idx_t* ids,
                               const uint8_t* codes) {
    if (n_entry == 0) {
        return;
    }

    LockGuard guard(lock_);

    // Critical section - modify data structures
    doc_ids_.insert(doc_ids_.end(), ids, ids + n_entry);
    codes_.insert(codes_.end(), codes, codes + (n_entry * element_size_));
}

void InvertedList::clear() {
    doc_ids_.clear();
    doc_ids_.shrink_to_fit();
    codes_.clear();
    codes_.shrink_to_fit();
}

void InvertedList::rebuild_from_entries(const std::vector<idx_t>& doc_ids,
                                        const std::vector<float>& scores) {
    LockGuard guard(lock_);

    doc_ids_ = doc_ids;
    codes_.resize(scores.size() * element_size_);
    std::memcpy(codes_.data(), scores.data(), scores.size() * sizeof(float));
}

std::vector<idx_t> InvertedList::prune_and_keep_doc_ids(size_t lambda) {
    LockGuard guard(lock_);

    size_t n_docs = doc_ids_.size();
    if (lambda <= 0 || n_docs == 0 || lambda >= n_docs) {
        return doc_ids_;
    }

    // Create pairs of (float_value, index) for sorting
    std::vector<std::pair<float, idx_t>> value_doc_pairs;
    value_doc_pairs.reserve(n_docs);

    for (size_t i = 0; i < n_docs; ++i) {
        value_doc_pairs.emplace_back(((float*)codes_.data())[i], doc_ids_[i]);
    }

    // Sort by float value in descending order (highest first)
    std::ranges::sort(value_doc_pairs, [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    std::vector<idx_t> kept_doc_ids;
    kept_doc_ids.reserve(lambda);
    std::transform(value_doc_pairs.begin(), value_doc_pairs.begin() + lambda,
                   std::back_inserter(kept_doc_ids),
                   [](const auto& pair) { return pair.second; });
    return kept_doc_ids;
}

InvertedLists::InvertedLists(size_t n_term, size_t element_size)
    : n_term_(n_term), element_size_(element_size) {}

void InvertedLists::add_entry(term_t term_id, idx_t doc_id,
                              const uint8_t* code) {
    add_entries(term_id, 1, &doc_id, code);
}

ArrayInvertedLists::ArrayInvertedLists(size_t n_term, size_t element_size)
    : InvertedLists(n_term, element_size) {
    lists_.reserve(n_term);
    for (size_t i = 0; i < n_term; ++i) {
        lists_.emplace_back(element_size);
    }
}

void ArrayInvertedLists::add_entries(term_t term_id, size_t n_entry,
                                     idx_t* doc_ids, const uint8_t* code) {
    if (term_id >= get_n_term()) {
        throw std::invalid_argument("term_id out of range");
    }
    auto& inverted_list = lists_[term_id];
    if (inverted_list.size() == 0) {
        ++actual_posting_list_size_;
    }
    inverted_list.add_entries(n_entry, doc_ids, code);
}

void ArrayInvertedLists::global_threshold_prune(size_t n_postings_per_list) {
    if (n_postings_per_list == 0) {
        return;
    }

    const size_t tot_postings = actual_posting_list_size_ * n_postings_per_list;
    constexpr size_t kEqualityThreshold = 10;
    const size_t max_eq_postings = kEqualityThreshold * tot_postings / 100;

    // Collect all postings: (score, doc_id, term_id)
    using posting_t = std::tuple<float, idx_t, term_t>;
    std::vector<posting_t> postings;

    size_t total_size = 0;
    for (const auto& list : lists_) {
        total_size += list.get_doc_ids().size();
    }
    postings.reserve(total_size);

    for (size_t term_id = 0; term_id < lists_.size(); ++term_id) {
        const auto& doc_ids = lists_[term_id].get_doc_ids();
        const auto& codes = lists_[term_id].get_codes();
        const auto* scores = reinterpret_cast<const float*>(codes.data());

        for (size_t idx = 0; idx < doc_ids.size(); ++idx) {
            postings.emplace_back(scores[idx], doc_ids[idx],
                                  static_cast<term_t>(term_id));
        }
        lists_[term_id].clear();
    }

    if (postings.empty()) {
        return;
    }

    // If we want more postings than we have, keep everything
    if (tot_postings >= postings.size()) {
        for (const auto& [score, doc_id, term_id] : postings) {
            float score_copy = score;
            lists_[term_id].add_entries(
                1, &doc_id, reinterpret_cast<const uint8_t*>(&score_copy));
        }
        return;
    }

    // Partial sort to find threshold - descending by score, then ascending by
    // doc_id
    std::nth_element(postings.begin(),
                     postings.begin() + static_cast<ptrdiff_t>(tot_postings),
                     postings.end(), [](const auto& lhs, const auto& rhs) {
                         if (std::get<0>(lhs) != std::get<0>(rhs)) {
                             return std::get<0>(lhs) > std::get<0>(rhs);
                         }
                         return std::get<1>(lhs) < std::get<1>(rhs);
                     });

    const float threshold_score = std::get<0>(postings[tot_postings]);

    // Collect postings with score equal to threshold (beyond tot_postings)
    std::vector<posting_t> eq_postings;
    for (size_t idx = tot_postings; idx < postings.size(); ++idx) {
        if (std::get<0>(postings[idx]) == threshold_score) {
            eq_postings.push_back(postings[idx]);
        }
    }

    // Temporary storage per term for rebuilding
    std::vector<std::vector<idx_t>> term_doc_ids(lists_.size());
    std::vector<std::vector<float>> term_scores(lists_.size());

    // Add postings above threshold
    for (size_t idx = 0; idx < tot_postings; ++idx) {
        const auto& [score, doc_id, term_id] = postings[idx];
        term_doc_ids[term_id].push_back(doc_id);
        term_scores[term_id].push_back(score);
    }

    // Add some equal-threshold postings (up to max_eq_postings)
    const size_t eq_to_add = std::min(max_eq_postings, eq_postings.size());
    for (size_t idx = 0; idx < eq_to_add; ++idx) {
        const auto& [score, doc_id, term_id] = eq_postings[idx];
        term_doc_ids[term_id].push_back(doc_id);
        term_scores[term_id].push_back(score);
    }

    if (eq_postings.size() > max_eq_postings) {
        std::cout
            << "GlobalThresholdPrune: "
            << (eq_postings.size() - max_eq_postings)
            << " entries with threshold score pruned due to equality limit\n";
    }

    // Rebuild each list sorted by score descending
    for (size_t term_id = 0; term_id < lists_.size(); ++term_id) {
        auto& doc_ids = term_doc_ids[term_id];
        auto& scores = term_scores[term_id];

        if (doc_ids.empty()) {
            continue;
        }

        // Sort by score descending
        std::vector<size_t> indices(doc_ids.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&scores](size_t lhs, size_t rhs) {
                      return scores[lhs] > scores[rhs];
                  });

        std::vector<idx_t> sorted_doc_ids(doc_ids.size());
        std::vector<float> sorted_scores(scores.size());
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            sorted_doc_ids[idx] = doc_ids[indices[idx]];
            sorted_scores[idx] = scores[indices[idx]];
        }

        lists_[term_id].rebuild_from_entries(sorted_doc_ids, sorted_scores);
    }
}

}  // namespace amaiss