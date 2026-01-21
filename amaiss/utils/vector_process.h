#ifndef VECTOR_PROCESS_H
#define VECTOR_PROCESS_H
#include <algorithm>
#include <vector>

#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"
#include "amaiss/utils/ranker.h"

namespace amaiss {
inline std::vector<term_t> top_k_tokens(SparseVectorView vec_view, int k) {
    TopKHolder<term_t> holder(k);
    const auto& [indices, weights] = vec_view;
    for (int i = 0; i < indices.size(); ++i) {
        holder.Add(weights[i], indices[i]);
    }
    auto top_k = holder.TopK();
    std::ranges::reverse(top_k);
    return top_k;
}

}  // namespace amaiss

#endif  // VECTOR_PROCESS_H