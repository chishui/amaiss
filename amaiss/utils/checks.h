#ifndef COMMON_H
#define COMMON_H

#include <stdexcept>

namespace amaiss {

template <typename T>
T* throw_if_null(T* ptr, const char* msg = "unexpected nullptr") {
    if (ptr == nullptr) {
        throw std::invalid_argument(msg);
    }
    return ptr;
}

template <typename T>
T throw_if_not_positive(T value, const char* msg = "value must be positive") {
    if (value <= 0) {
        throw std::invalid_argument(msg);
    }
    return value;
}

template <typename... Args>
void throw_if_any_null(const char* msg, Args*... ptrs) {
    bool any_null = ((ptrs == nullptr) || ...);
    if (any_null) {
        throw std::invalid_argument(msg);
    }
}

template <typename... Args>
void throw_if_any_null(Args*... ptrs) {
    throw_if_any_null("unexpected nullptr", ptrs...);
}

}  // namespace amaiss

#endif  // COMMON_H