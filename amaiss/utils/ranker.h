#ifndef RANKER_H
#define RANKER_H

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "amaiss/utils/checks.h"

namespace amaiss {

template <typename T, typename Comparator = std::greater<float>>
class TopKHolder {
    using P = std::pair<float, T>;
    int k;
    struct CompareP {
        Comparator comp;
        bool operator()(const P& p1, const P& p2) const {
            return comp(p1.first, p2.first);
        }
    };

    std::priority_queue<P, std::vector<P>, CompareP> pq;
    float threshold_ = std::numeric_limits<float>::infinity();

public:
    TopKHolder(int k) : k(k) {
        throw_if_not_positive(k);
        std::vector<P> vec;
        vec.reserve(k);
        pq = std::priority_queue<P, std::vector<P>, CompareP>(CompareP(),
                                                              std::move(vec));
    }

    // use a priority_queue to hold the top K items with highest scores
    void add(const float score, const T& item) {
        if (pq.size() >= k && score <= threshold_) {
            return;  // Fast reject without touching heap
        }
        if (pq.size() < k) {
            pq.emplace(score, item);
            if (pq.size() == k) threshold_ = pq.top().first;
        } else {
            pq.pop();
            pq.emplace(score, item);
            threshold_ = pq.top().first;
        }
    }
    /**
     * get k data from pq in value ascending order, this is a disruptive
     * operation
     */
    std::vector<T> top_k() {
        if (k <= 0) return {};
        std::vector<T> ret(k);
        int idx = 0;
        while (!pq.empty() && idx < k) {
            ret[idx] = pq.top().second;
            pq.pop();
            ++idx;
        }
        return ret;
    }

    /**
     *  get data from pq in value descending order (highest scores first),
     *  this is a disruptive operation. Always returns exactly k elements,
     *  padding with default-constructed T if pq has fewer than k items.
     */
    std::vector<T> top_k_descending() {
        size_t size = pq.size();
        if (k <= 0 || size <= 0) return {};
        std::vector<T> ret(size);
        int idx = size - 1;
        while (!pq.empty() && idx < k) {
            ret[idx--] = pq.top().second;
            pq.pop();
        }
        return ret;
    }

    std::vector<T> top_k_descending_with_padding(T pad_with) {
        std::vector<T> ret = top_k_descending();
        ret.resize(k, pad_with);
        return ret;
    }

    [[nodiscard]] bool full() { return pq.size() == k; }
    [[nodiscard]] bool empty() { return pq.empty(); }

    size_t size() { return pq.size(); }

    float peek_score() { return pq.top().first; }
};

template <typename T, typename ID_T = size_t,
          typename Comparator = std::greater<float>>
class DedupeTopKHolder {
    using P = std::pair<float, std::pair<ID_T, T>>;

private:
    int k;
    struct CompareP {
        Comparator comp;
        bool operator()(const P& p1, const P& p2) const {
            return comp(p1.first, p2.first);
        }
    };

    std::priority_queue<P, std::vector<P>, CompareP> pq;
    absl::flat_hash_set<ID_T> dedupe;

public:
    DedupeTopKHolder(int k) : k(k) {
        dedupe.reserve(k);
        std::vector<P> vec;
        vec.reserve(k);
        pq = std::priority_queue<P, std::vector<P>, CompareP>(CompareP(),
                                                              std::move(vec));
    }

    // use a priority_queue to hold the top K items with highest scores
    void add(const float score, ID_T id, const T& item) {
        if (pq.size() >= k && score <= pq.top().first) {
            return;
        }
        if (dedupe.find(id) != dedupe.end()) {
            return;
        }
        if (pq.size() < k) {
            pq.emplace(std::make_pair(score, std::make_pair(id, item)));
            dedupe.insert(id);
        } else if (pq.top().first < score) {
            auto top = pq.top();
            dedupe.erase(top.second.first);
            pq.pop();
            pq.emplace(std::make_pair(score, std::make_pair(id, item)));
            dedupe.insert(id);
        }
    }

    void add(const float score, ID_T id) {
        if (pq.size() >= k && score <= pq.top().first) {
            return;
        }
        if (dedupe.find(id) != dedupe.end()) {
            return;
        }
        if (pq.size() < k) {
            pq.push({score, {id, id}});
            dedupe.insert(id);
        } else if (pq.top().first < score) {
            auto top = pq.top();
            dedupe.erase(top.second.first);
            pq.pop();
            pq.push({score, {id, id}});
            dedupe.insert(id);
        }
    }

    [[nodiscard]] bool full() { return pq.size() == k; }

    /**
     *  get data from pq, this is a disruptive operation
     */
    std::vector<T> top_k() {
        std::vector<T> ret;
        ret.reserve(pq.size());
        while (!pq.empty()) {
            ret.push_back(pq.top().second.second);
            pq.pop();
        }
        return ret;
    }

    /**
     *  get data from pq in descending order (highest scores first),
     *  this is a disruptive operation. Always returns exactly k elements,
     *  padding with default-constructed T if pq has fewer than k items.
     */
    std::vector<T> top_k_descending() {
        size_t size = pq.size();
        if (k <= 0 || size <= 0) return {};
        std::vector<T> ret(size);
        int idx = size - 1;
        while (!pq.empty() && idx < k) {
            ret[idx--] = pq.top().second.second;
            pq.pop();
        }
        return ret;
    }

    std::vector<T> top_k_descending_with_padding(T pad_with) {
        std::vector<T> ret = top_k_descending();
        ret.resize(k, pad_with);
        return ret;
    }

    bool empty() { return pq.empty(); }

    size_t size() { return pq.size(); }

    float peek_score() { return pq.top().first; }
};

}  // namespace amaiss
#endif