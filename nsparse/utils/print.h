#ifndef PRINT_H
#define PRINT_H

#include <format>
#include <iostream>
#include <vector>

template <typename T>
struct std::formatter<std::vector<T>> {
    // Parse format specifier (optional, can be empty)
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();  // No custom format specifiers
    }

    // Format the vector
    auto format(const std::vector<T>& vec, std::format_context& ctx) const {
        auto out = ctx.out();
        *out++ = '[';

        bool first = true;
        for (const auto& elem : vec) {
            if (!first) {
                *out++ = ',';
                *out++ = ' ';
            }
            out = std::format_to(out, "{}", elem);
            first = false;
        }

        *out++ = ']';
        return out;
    }
};

#endif  // PRINT_H