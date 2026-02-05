#ifndef VECTOR_PROCESS_H
#define VECTOR_PROCESS_H
#include <algorithm>
#include <vector>

#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"
#include "amaiss/utils/ranker.h"

namespace amaiss {
template <class T>
inline std::vector<term_t> top_k_tokens(const term_t* indices, const T* weights,
                                        int size, int k) {
    TopKHolder<term_t> holder(k);
    for (int i = 0; i < size; ++i) {
        holder.add(weights[i], indices[i]);
    }
    return holder.top_k_descending();
}

}  // namespace amaiss

#endif  // VECTOR_PROCESS_H