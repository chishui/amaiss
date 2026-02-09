#ifndef INDEX_COMMON_H
#define INDEX_COMMON_H
#include <utility>
#include <vector>

#include "amaiss/types.h"
#include "amaiss/utils/ranker.h"

namespace amaiss {
namespace detail {
auto top_k_results_to_query_results(const std::vector<TopKItem<idx_t>>& results,
                                    int k)
    -> std::pair<std::vector<float>, std::vector<idx_t>>;
}  // namespace detail
}  // namespace amaiss

#endif  // INDEX_COMMON_H