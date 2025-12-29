#pragma once

#include "Types.h"
#include <cmath>
#include <array>

namespace intgr {

/// Fixed-point mathematical functions using lookup tables
/// All functions operate on int8 inputs and produce scaled int outputs
namespace FixedMath {

/// Sigmoid lookup table (256 entries)
/// Maps int8 [-128, 127] to sigmoid output scaled to [0, 255]
/// sigmoid(x) ≈ 1 / (1 + exp(-x))
struct SigmoidLUT {
    std::array<u8, 256> table;

    SigmoidLUT() : table{} {
        // Generate sigmoid values
        // For int8 input x in [-128, 127], map to float range [-8, 8] (reasonable sigmoid domain)
        // Then scale sigmoid(x) from [0, 1] to [0, 255]
        for (usize i = 0; i < 256; ++i) {
            i8 x = static_cast<i8>(static_cast<i32>(i) - constants::I8_OFFSET);

            // Map [-128, 127] to [-8, 8] (divide by SIGMOID_DOMAIN_SCALE)
            f32 x_float = static_cast<f32>(x) / static_cast<f32>(constants::SIGMOID_DOMAIN_SCALE);

            // Compute sigmoid: 1 / (1 + exp(-x))
            f32 sigmoid_val = 1.0f / (1.0f + std::exp(-x_float));

            // Scale to [0, 255]
            u8 scaled = static_cast<u8>(sigmoid_val * 255.0f + 0.5f);
            table[i] = scaled;
        }
    }

    /// Lookup sigmoid value for int8 input
    /// @param x  Input in range [-128, 127]
    /// @return Sigmoid output scaled to [0, 255]
    u8 operator()(i8 x) const noexcept {
        return table[static_cast<u8>(x + constants::I8_OFFSET)];
    }

    /// Compute sigmoid for int32 value (with clamping)
    /// @param x  Input int32
    /// @return Sigmoid output scaled to [0, 255]
    u8 sigmoid_i32(i32 x) const noexcept {
        // Clamp to int8 range
        if (x < -128) return table[0];
        if (x > 127) return table[255];
        return table[static_cast<u8>(x + constants::I8_OFFSET)];
    }
};

/// Global sigmoid lookup table (initialized at runtime)
inline const SigmoidLUT sigmoid_lut;

/// Apply sigmoid to int8 value
/// @param x  Input in [-128, 127]
/// @return Sigmoid(x) scaled to [0, 255]
inline u8 sigmoid(i8 x) noexcept {
    return sigmoid_lut(x);
}

/// Apply sigmoid to int32 value (with clamping)
/// @param x  Input int32
/// @return Sigmoid(x) scaled to [0, 255]
inline u8 sigmoid(i32 x) noexcept {
    return sigmoid_lut.sigmoid_i32(x);
}

/// Log2 lookup table for unsigned integers
/// Maps u8 [1, 255] to log2(x) scaled by 256 (Q8.8 fixed-point)
struct Log2LUT {
    std::array<i16, 256> table;

    Log2LUT() : table{} {
        table[0] = -32768;  // log2(0) = -inf, use min value

        for (usize i = 1; i < 256; ++i) {
            // Compute log2(i) and scale by 256 (Q8.8 format)
            f32 log_val = std::log2(static_cast<f32>(i));
            table[i] = static_cast<i16>(log_val * 256.0f);
        }
    }

    /// Lookup log2 value for u8 input
    /// @param x  Input in range [0, 255]
    /// @return log2(x) in Q8.8 fixed-point (scaled by 256)
    i16 operator()(u8 x) const noexcept {
        return table[x];
    }
};

/// Global log2 lookup table (initialized at runtime)
inline const Log2LUT log2_lut;

/// Compute log2 of unsigned byte
/// @param x  Input in [0, 255]
/// @return log2(x) in Q8.8 fixed-point
inline i16 log2(u8 x) noexcept {
    return log2_lut(x);
}

/// Fast integer square root using Newton's method
/// @param x  Input (non-negative)
/// @return floor(sqrt(x))
inline u32 sqrt_u32(u32 x) noexcept {
    if (x == 0) return 0;
    if (x == 1) return 1;

    // Digit-by-digit calculation
    u32 result = 0;
    u32 bit = 1u << 30;  // Second-to-top bit set

    // Find the highest set bit
    while (bit > x) {
        bit >>= 2;
    }

    // Calculate sqrt digit by digit
    while (bit != 0) {
        if (x >= result + bit) {
            x -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }

    return result;
}

/// Fast integer square root for int64
/// @param x  Input (non-negative)
/// @return floor(sqrt(x))
inline u32 sqrt_i64(i64 x) noexcept {
    if (x < 0) return 0;
    if (x <= 0xFFFFFFFF) {
        return sqrt_u32(static_cast<u32>(x));
    }

    // For large values, use shift approximation
    u64 result = 0;
    u64 bit = 1ULL << 62;

    u64 n = static_cast<u64>(x);

    while (bit > n) {
        bit >>= 2;
    }

    while (bit != 0) {
        if (n >= result + bit) {
            n -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }

    return static_cast<u32>(result);
}

} // namespace FixedMath

} // namespace intgr
