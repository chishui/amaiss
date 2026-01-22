#ifndef PREFETCH_H
#define PREFETCH_H
#include "amaiss/sparse_vectors.h"

namespace amaiss {

inline void prefetch_next_vector(const SparseVectors* vectors, idx_t doc_id) {
    static constexpr size_t kCacheLineSize = 64;  // bytes
    const auto& view = vectors->get_vector_view(doc_id);
    const size_t indices_bytes = view.indices.size() * sizeof(view.indices[0]);
    const size_t values_bytes = view.values.size() * sizeof(view.values[0]);

    const char* indices_ptr =
        reinterpret_cast<const char*>(view.indices.data());
    const char* values_ptr = reinterpret_cast<const char*>(view.values.data());

    for (size_t offset = 0; offset < indices_bytes; offset += kCacheLineSize) {
        __builtin_prefetch(indices_ptr + offset, 0, 0);
    }
    for (size_t offset = 0; offset < values_bytes; offset += kCacheLineSize) {
        __builtin_prefetch(values_ptr + offset, 0, 0);
    }
}
}  // namespace amaiss

#endif  // PREFETCH_H