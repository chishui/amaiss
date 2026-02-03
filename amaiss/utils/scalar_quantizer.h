#ifndef SCALAR_QUANTIZER_H
#define SCALAR_QUANTIZER_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace amaiss {

/// Quantization type for scalar quantizer
enum class QuantizerType : uint8_t {
    QT_8bit,   // 8-bit quantization
    QT_16bit,  // 16-bit quantization
};

/// Scalar Quantizer using min-max quantization for sparse vector values
/// Quantizes individual float values to 8-bit or 16-bit integers
class ScalarQuantizer {
public:
    ScalarQuantizer() : qtype_(QuantizerType::QT_8bit), vmin_(0), vmax_(1) {}

    ScalarQuantizer(QuantizerType qtype, float vmin, float vmax)
        : qtype_(qtype), vmin_(vmin), vmax_(vmax) {
        if (vmax <= vmin) {
            throw std::invalid_argument("vmax must be greater than vmin");
        }
    }

    /// Returns bytes per quantized value
    [[nodiscard]] size_t bytes_per_value() const {
        return (qtype_ == QuantizerType::QT_8bit) ? 1 : 2;
    }

    /// Encode array of values
    void encode(const float* vals, uint8_t* codes, size_t n) const {
        if (qtype_ == QuantizerType::QT_8bit) {
            for (size_t i = 0; i < n; i++) {
                codes[i] = encode_8bit(vals[i]);
            }
        } else {
            auto* codes16 = reinterpret_cast<uint16_t*>(codes);
            for (size_t i = 0; i < n; i++) {
                codes16[i] = encode_16bit(vals[i]);
            }
        }
    }

    /// Decode array of values
    void decode(const uint8_t* codes, float* vals, size_t n) const {
        if (qtype_ == QuantizerType::QT_8bit) {
            for (size_t i = 0; i < n; i++) {
                vals[i] = decode_8bit(codes[i]);
            }
        } else {
            const auto* codes16 = reinterpret_cast<const uint16_t*>(codes);
            for (size_t i = 0; i < n; i++) {
                vals[i] = decode_16bit(codes16[i]);
            }
        }
    }

    QuantizerType get_quantizer_type() const { return qtype_; }

private:
    /// Encode a single float value to 8-bit
    [[nodiscard]] uint8_t encode_8bit(float val) const {
        float scaled = (val - vmin_) * (kMax8bit / (vmax_ - vmin_));
        scaled = std::max(0.0F, std::min(kMax8bit, scaled));
        return static_cast<uint8_t>(std::lround(scaled));
    }

    /// Decode 8-bit back to float
    [[nodiscard]] float decode_8bit(uint8_t code) const {
        return vmin_ + (static_cast<float>(code) * (vmax_ - vmin_) / kMax8bit);
    }

    /// Encode a single float value to 16-bit
    [[nodiscard]] uint16_t encode_16bit(float val) const {
        float scaled = (val - vmin_) * (kMax16bit / (vmax_ - vmin_));
        scaled = std::max(0.0F, std::min(kMax16bit, scaled));
        return static_cast<uint16_t>(std::lround(scaled));
    }

    /// Decode 16-bit back to float
    [[nodiscard]] float decode_16bit(uint16_t code) const {
        return vmin_ + (static_cast<float>(code) * (vmax_ - vmin_) / kMax16bit);
    }

    static constexpr float kMax8bit = std::numeric_limits<uint8_t>::max();
    static constexpr float kMax16bit = std::numeric_limits<uint16_t>::max();

    QuantizerType qtype_;
    float vmin_;  // minimum value for quantization range
    float vmax_;  // maximum value for quantization range
};

}  // namespace amaiss

#endif  // SCALAR_QUANTIZER_H
