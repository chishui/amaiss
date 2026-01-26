#ifndef PREFETCH_H
#define PREFETCH_H
#include <cstddef>

#include "amaiss/types.h"

namespace amaiss {

inline void prefetch_vector(const term_t* indices, const float* values,
                            size_t len) {
    static constexpr size_t kCacheLineSize = 64;  // bytes

    const char* indices_ptr = reinterpret_cast<const char*>(indices);
    const char* values_ptr = reinterpret_cast<const char*>(values);

    const size_t indices_bytes = len * sizeof(term_t);
    const size_t values_bytes = len * sizeof(float);

    for (size_t offset = 0; offset < indices_bytes; offset += kCacheLineSize) {
        __builtin_prefetch(indices_ptr + offset, 0, 0);
    }
    for (size_t offset = 0; offset < values_bytes; offset += kCacheLineSize) {
        __builtin_prefetch(values_ptr + offset, 0, 0);
    }
}

}  // namespace amaiss

#endif  // PREFETCH_H