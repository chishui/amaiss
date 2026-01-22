#include "amaiss/invlists/inverted_lists.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
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
    inverted_list.add_entries(n_entry, doc_ids, code);
}

}  // namespace amaiss