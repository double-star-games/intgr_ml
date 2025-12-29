#pragma once

#include "Types.h"

// SIMD abstraction layer for IntgrML
// Provides compile-time selection between AVX2, NEON, and scalar backends
//
// Usage:
//   - Include this header
//   - Check SNAP_SIMD_AVX2, SNAP_SIMD_NEON, or SNAP_SIMD_NONE
//   - Use intrinsics directly when available
//   - Fall back to scalar for unsupported platforms

// Detect SIMD support at compile time
#if defined(__AVX2__) && !defined(SNAP_DISABLE_SIMD)
    #define SNAP_SIMD_AVX2 1
    #include <immintrin.h>
#elif defined(__ARM_NEON) && !defined(SNAP_DISABLE_SIMD)
    #define SNAP_SIMD_NEON 1
    #include <arm_neon.h>
#else
    #define SNAP_SIMD_NONE 1
#endif

namespace intgr {
namespace simd {

/// Get SIMD backend name (for logging/debugging)
inline const char* backend_name() {
#if defined(SNAP_SIMD_AVX2)
    return "AVX2";
#elif defined(SNAP_SIMD_NEON)
    return "NEON";
#else
    return "Scalar";
#endif
}

/// Check if SIMD is enabled
inline constexpr bool is_enabled() {
#if defined(SNAP_SIMD_AVX2) || defined(SNAP_SIMD_NEON)
    return true;
#else
    return false;
#endif
}

/// Get SIMD vector width in bytes
inline constexpr usize vector_width_bytes() {
#if defined(SNAP_SIMD_AVX2)
    return 32;  // 256-bit AVX2
#elif defined(SNAP_SIMD_NEON)
    return 16;  // 128-bit NEON
#else
    return 1;   // Scalar
#endif
}

/// Get SIMD vector width in int8 elements
inline constexpr usize vector_width_i8() {
#if defined(SNAP_SIMD_AVX2)
    return 32;  // 32 × int8
#elif defined(SNAP_SIMD_NEON)
    return 16;  // 16 × int8
#else
    return 1;
#endif
}

/// Get SIMD vector width in int32 elements
inline constexpr usize vector_width_i32() {
#if defined(SNAP_SIMD_AVX2)
    return 8;   // 8 × int32
#elif defined(SNAP_SIMD_NEON)
    return 4;   // 4 × int32
#else
    return 1;
#endif
}

//==============================================================================
// Horizontal sum operations
//==============================================================================

/// Horizontal sum of 8 × int32 (AVX2 only)
#if defined(SNAP_SIMD_AVX2)
inline i32 horizontal_sum_i32(__m256i vec) {
    // Sum upper and lower 128-bit halves
    __m128i lo = _mm256_castsi256_si128(vec);
    __m128i hi = _mm256_extracti128_si256(vec, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);

    // Horizontal sum within 128 bits
    __m128i sum64 = _mm_hadd_epi32(sum128, sum128);
    __m128i sum32 = _mm_hadd_epi32(sum64, sum64);

    return _mm_cvtsi128_si32(sum32);
}
#endif

/// Horizontal sum of 4 × int32 (NEON only)
#if defined(SNAP_SIMD_NEON)
inline i32 horizontal_sum_i32(int32x4_t vec) {
    // Pairwise addition twice
    int32x2_t sum_pair = vadd_s32(vget_low_s32(vec), vget_high_s32(vec));
    return vget_lane_s32(vpadd_s32(sum_pair, sum_pair), 0);
}
#endif

//==============================================================================
// Memory operations
//==============================================================================

/// Aligned allocation (32-byte for AVX2, 16-byte for NEON)
inline void* aligned_alloc(usize size) {
    constexpr usize alignment =
#if defined(SNAP_SIMD_AVX2)
        32;
#elif defined(SNAP_SIMD_NEON)
        16;
#else
        8;
#endif

#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

/// Aligned free
inline void aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

//==============================================================================
// Histogram operations (used in gradient boosting)
//==============================================================================

/// Add multiple samples to histogram bins (vectorized)
/// @param features  Array of int8 feature values
/// @param gradients  Array of int32 gradients
/// @param hessians  Array of int32 hessians
/// @param count  Number of samples
/// @param bin_counts  Histogram bin counts (256 bins)
/// @param bin_grads  Histogram bin gradient sums (256 bins)
/// @param bin_hess  Histogram bin hessian sums (256 bins)
inline void histogram_add_batch(
    const i8* features,
    const i32* gradients,
    const i32* hessians,
    usize count,
    i32* bin_counts,
    i32* bin_grads,
    i32* bin_hess
) {
    // Scalar fallback for now - will be vectorized in separate implementation
    for (usize i = 0; i < count; ++i) {
        usize bin_idx = static_cast<usize>(static_cast<i32>(features[i]) + constants::I8_OFFSET);
        bin_counts[bin_idx]++;
        bin_grads[bin_idx] += gradients[i];
        bin_hess[bin_idx] += hessians[i];
    }
}

} // namespace simd
} // namespace intgr
