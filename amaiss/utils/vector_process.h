#ifndef VECTOR_PROCESS_H
#define VECTOR_PROCESS_H
#include <vector>

#include "amaiss/types.h"
#include "amaiss/utils/ranker.h"

namespace amaiss::detail {
template <class T>
inline std::vector<term_t> top_k_tokens(const term_t* indices, const T* weights,
                                        int size, int k) {
    if (k >= size) {
        std::vector<term_t> result(indices, indices + size);
        return result;
    }
    TopKHolder<term_t> holder(k);
    for (int i = 0; i < size; ++i) {
        holder.add(weights[i], indices[i]);
    }
    return holder.top_k_descending();
}

}  // namespace amaiss::detail

#endif  // VECTOR_PROCESS_H