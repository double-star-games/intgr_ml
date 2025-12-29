#pragma once

#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace intgr {

// Integer types (explicit sizing)
using i8  = int8_t;
using u8  = uint8_t;
using i16 = int16_t;
using u16 = uint16_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;

// Floating point (only for quantization, not inference)
using f32 = float;
using f64 = double;

// Size types
using usize = std::size_t;

// Fixed-point and quantization constants
namespace constants {
    /// Q8.8 fixed-point scale factor (8 bits fractional precision)
    /// Used for learning rates, scaling factors, and probability values
    constexpr i32 FIXED_POINT_SCALE = 256;

    /// Number of bins for quantized histograms and features
    /// Maps int8 range [-128, 127] to bin indices [0, 255]
    constexpr usize HISTOGRAM_BINS = 256;

    /// Offset to convert int8 values to bin indices
    /// bin_index = int8_value + I8_OFFSET (e.g., -128 -> 0, 127 -> 255)
    constexpr i32 I8_OFFSET = 128;

    /// Sigmoid lookup table domain scaling factor
    /// Determines resolution of sigmoid approximation
    constexpr i32 SIGMOID_DOMAIN_SCALE = 16;
}

// Overflow handling modes
enum class OverflowMode : u8 {
    SATURATE,  // Clamp to type limits (production default)
    TRAP,      // Assert/abort on overflow (debug default)
    WRAP       // Undefined behavior, fastest (use with caution)
};

// Compile-time overflow mode selection
#ifndef SNAPML_OVERFLOW_MODE
    #ifdef NDEBUG
        #define SNAPML_OVERFLOW_MODE SATURATE
    #else
        #define SNAPML_OVERFLOW_MODE TRAP
    #endif
#endif

// Saturating arithmetic helpers
namespace detail {
    template<typename T>
    constexpr T sat_add(T a, T b) noexcept {
        static_assert(std::is_signed_v<T>, "saturating add requires signed type");

        // Check for positive overflow
        if (b > 0 && a > std::numeric_limits<T>::max() - b) {
            return std::numeric_limits<T>::max();
        }
        // Check for negative overflow
        if (b < 0 && a < std::numeric_limits<T>::min() - b) {
            return std::numeric_limits<T>::min();
        }
        return static_cast<T>(a + b);
    }

    template<typename T>
    constexpr T sat_sub(T a, T b) noexcept {
        static_assert(std::is_signed_v<T>, "saturating sub requires signed type");

        // Check for negative overflow (underflow)
        if (b > 0 && a < std::numeric_limits<T>::min() + b) {
            return std::numeric_limits<T>::min();
        }
        // Check for positive overflow
        if (b < 0 && a > std::numeric_limits<T>::max() + b) {
            return std::numeric_limits<T>::max();
        }
        return static_cast<T>(a - b);
    }

    template<typename T>
    constexpr T sat_mul(T a, T b) noexcept {
        static_assert(std::is_signed_v<T>, "saturating mul requires signed type");

        // Promote to larger type for overflow detection
        using Wider = std::conditional_t<sizeof(T) <= 4, i64, i64>;
        Wider result = static_cast<Wider>(a) * static_cast<Wider>(b);

        if (result > std::numeric_limits<T>::max()) {
            return std::numeric_limits<T>::max();
        }
        if (result < std::numeric_limits<T>::min()) {
            return std::numeric_limits<T>::min();
        }
        return static_cast<T>(result);
    }
}

// Safe arithmetic macros (behavior depends on SNAPML_OVERFLOW_MODE)
#if SNAPML_OVERFLOW_MODE == 0  // SATURATE
    #define SNAP_ADD(a, b) (::intgr::detail::sat_add((a), (b)))
    #define SNAP_SUB(a, b) (::intgr::detail::sat_sub((a), (b)))
    #define SNAP_MUL(a, b) (::intgr::detail::sat_mul((a), (b)))
#elif SNAPML_OVERFLOW_MODE == 1  // TRAP
    #include <cassert>
    #define SNAP_ADD(a, b) ([&](){ \
        auto _a = (a), _b = (b); \
        assert((_b > 0 && _a <= std::numeric_limits<decltype(_a)>::max() - _b) || \
               (_b <= 0 && _a >= std::numeric_limits<decltype(_a)>::min() - _b)); \
        return _a + _b; \
    }())
    #define SNAP_SUB(a, b) ([&](){ \
        auto _a = (a), _b = (b); \
        assert((_b < 0 && _a <= std::numeric_limits<decltype(_a)>::max() + _b) || \
               (_b >= 0 && _a >= std::numeric_limits<decltype(_a)>::min() + _b)); \
        return _a - _b; \
    }())
    #define SNAP_MUL(a, b) ([&](){ \
        auto _a = (a), _b = (b); \
        using W = std::conditional_t<sizeof(decltype(_a)) <= 4, i64, i64>; \
        W _r = static_cast<W>(_a) * static_cast<W>(_b); \
        assert(_r >= std::numeric_limits<decltype(_a)>::min() && \
               _r <= std::numeric_limits<decltype(_a)>::max()); \
        return static_cast<decltype(_a)>(_r); \
    }())
#else  // WRAP (unsafe, fast)
    #define SNAP_ADD(a, b) ((a) + (b))
    #define SNAP_SUB(a, b) ((a) - (b))
    #define SNAP_MUL(a, b) ((a) * (b))
#endif

// Clamp utility
template<typename T>
constexpr T clamp(T val, T min_val, T max_val) noexcept {
    return val < min_val ? min_val : (val > max_val ? max_val : val);
}

// Round division (for fixed-point scaling)
template<typename T>
constexpr T div_round(T numerator, T denominator) noexcept {
    static_assert(std::is_signed_v<T>, "div_round requires signed type");
    T q = numerator / denominator;
    T r = numerator % denominator;
    // Round to nearest (half away from zero)
    if (r * 2 >= denominator) return q + 1;
    if (r * 2 <= -denominator) return q - 1;
    return q;
}

} // namespace intgr
