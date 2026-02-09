#include "amaiss/index_common.h"

#include <utility>
#include <vector>

#include "amaiss/types.h"
#include "amaiss/utils/ranker.h"
namespace amaiss {
namespace detail {
auto top_k_results_to_query_results(const std::vector<TopKItem<idx_t>>& results,
                                    int k)
    -> std::pair<std::vector<float>, std::vector<idx_t>> {
    std::vector<float> distances(k, -1.0F);
    std::vector<idx_t> labels(k, INVALID_IDX);
    if (results.empty()) {
        return {distances, labels};
    }
    for (int i = 0; i < results.size(); ++i) {
        const auto& [score, label] = results[results.size() - i - 1];
        distances[i] = score;
        labels[i] = label;
    }
    return {distances, labels};
}
}  // namespace detail
}  // namespace amaiss