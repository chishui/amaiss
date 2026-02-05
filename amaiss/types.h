#ifndef TYPES_H
#define TYPES_H

#include <cstdint>

namespace amaiss {

using idx_t = int32_t;
using term_t = uint16_t;
using weight_t = float;

constexpr idx_t INVALID_IDX = -1;
}  // namespace amaiss

#endif  // TYPES_H