#ifndef VECTOR_PROCESS_H
#define VECTOR_PROCESS_H
#include <algorithm>
#include <vector>

#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"
#include "amaiss/utils/ranker.h"

namespace amaiss {
inline std::vector<term_t> top_k_tokens(const term_t* indices,
                                        const float* weights, int size, int k) {
    TopKHolder<term_t> holder(k);
    for (int i = 0; i < size; ++i) {
        holder.add(weights[i], indices[i]);
    }
    auto top_k = holder.top_k();
    std::ranges::reverse(top_k);
    return top_k;
}

}  // namespace amaiss

#endif  // VECTOR_PROCESS_H