#ifndef PREFETCH_H
#define PREFETCH_H
#include <cstddef>

#include "amaiss/types.h"

#ifndef AMAISS_PREFETCH
#define AMAISS_PREFETCH(addr, rw, locality) \
    __builtin_prefetch(addr, rw, locality)
#endif

namespace amaiss::detail {

template <class T>
inline void prefetch_vector(const term_t* indices, const T* values,
                            size_t len) {
    static constexpr size_t kCacheLineSize = 64;  // bytes

    const char* indices_ptr = reinterpret_cast<const char*>(indices);
    const char* values_ptr = reinterpret_cast<const char*>(values);

    const size_t indices_bytes = len * sizeof(term_t);
    const size_t values_bytes = len * sizeof(T);

    for (size_t offset = 0; offset < indices_bytes; offset += kCacheLineSize) {
        AMAISS_PREFETCH(indices_ptr + offset, 0, 0);
    }
    for (size_t offset = 0; offset < values_bytes; offset += kCacheLineSize) {
        AMAISS_PREFETCH(values_ptr + offset, 0, 0);
    }
}

inline void prefetch_indptr(const idx_t* indptr, idx_t doc_id) {
    AMAISS_PREFETCH(&indptr[doc_id], 0, 0);
}

}  // namespace amaiss::detail

#endif  // PREFETCH_H