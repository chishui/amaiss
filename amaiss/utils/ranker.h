#ifndef RANKER_H
#define RANKER_H

#include <functional>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

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

public:
    TopKHolder(int k) : k(k) {
        std::vector<P> vec;
        vec.reserve(k);
        pq = std::priority_queue<P, std::vector<P>, CompareP>(CompareP(),
                                                              std::move(vec));
    }

    // use a priority_queue to hold the top K items with highest scores
    void Add(const float score, const T& item) {
        if (pq.size() < k) {
            pq.push(std::make_pair(score, item));
        } else if (pq.top().first < score) {
            pq.pop();
            pq.push(std::make_pair(score, item));
        }
    }

    void Add_simple(const float score, const T& item) {
        pq.push(std::make_pair(score, item));
    }

    void Pop_simple() { pq.pop(); }

    /**
     *  get data from pq, this is a disruptive operation
     */
    std::vector<T> TopK() {
        std::vector<T> ret;
        ret.reserve(pq.size());
        while (!pq.empty()) {
            ret.push_back(pq.top().second);
            pq.pop();
        }
        return ret;
    }

    bool empty() { return pq.empty(); }

    size_t size() { return pq.size(); }

    float PeekScore() { return pq.top().first; }
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
    std::unordered_set<ID_T> dedupe;

public:
    DedupeTopKHolder(int k) : k(k) {
        dedupe.reserve(k);
        std::vector<P> vec;
        vec.reserve(k);
        pq = std::priority_queue<P, std::vector<P>, CompareP>(CompareP(),
                                                              std::move(vec));
    }

    // use a priority_queue to hold the top K items with highest scores
    void Add(const float score, ID_T id, const T& item) {
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

    void Add(const float score, ID_T id) {
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

    bool IsFull() { return pq.size() == k; }

    /**
     *  get data from pq, this is a disruptive operation
     */
    std::vector<T> TopK() {
        std::vector<T> ret;
        ret.reserve(pq.size());
        while (!pq.empty()) {
            ret.push_back(pq.top().second.second);
            pq.pop();
        }
        return ret;
    }

    bool empty() { return pq.empty(); }

    size_t size() { return pq.size(); }

    float PeekScore() { return pq.top().first; }
};

}  // namespace amaiss
#endif