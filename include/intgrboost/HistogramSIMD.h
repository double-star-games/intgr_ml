#pragma once

#include "../intgrcore/Types.h"
#include "../intgrcore/SIMD.h"
#include "Histogram.h"

namespace intgr {

/// SIMD-accelerated histogram building for gradient boosting
/// Provides AVX2/NEON optimized implementations with fallback to scalar
class HistogramSIMD {
public:
    /// Build histogram with SIMD acceleration
    /// @param features  Feature values (int8 array)
    /// @param gradients  Gradient values (int32 array)
    /// @param hessians  Hessian values (int32 array)
    /// @param sample_mask  Mask indicating which samples to include (1 = include, 0 = skip)
    /// @param count  Number of samples
    /// @param hist  Output histogram (must be cleared before calling)
    static void build(
        const i8* features,
        const i32* gradients,
        const i32* hessians,
        const u8* sample_mask,
        usize count,
        Histogram& hist
    ) {
#if defined(SNAP_SIMD_AVX2)
        build_avx2(features, gradients, hessians, sample_mask, count, hist);
#elif defined(SNAP_SIMD_NEON)
        build_neon(features, gradients, hessians, sample_mask, count, hist);
#else
        build_scalar(features, gradients, hessians, sample_mask, count, hist);
#endif
    }

private:
    //==========================================================================
    // Scalar implementation (baseline)
    //==========================================================================

    static void build_scalar(
        const i8* features,
        const i32* gradients,
        const i32* hessians,
        const u8* sample_mask,
        usize count,
        Histogram& hist
    ) {
        for (usize i = 0; i < count; ++i) {
            if (sample_mask[i] == 0) continue;

            i8 feature_val = features[i];
            i32 gradient = gradients[i];
            i32 hessian = hessians[i];

            hist.add(feature_val, gradient, hessian);
        }
    }

    //==========================================================================
    // AVX2 implementation (x86_64)
    //==========================================================================

#if defined(SNAP_SIMD_AVX2)
    static void build_avx2(
        const i8* features,
        const i32* gradients,
        const i32* hessians,
        const u8* sample_mask,
        usize count,
        Histogram& hist
    ) {
        // Process 8 samples at a time using AVX2
        // Note: Histogram updates are not easily vectorizable due to random access
        // pattern, so we use SIMD for loading/masking but scalar for accumulation

        usize i = 0;
        const usize vec_width = 8;
        const usize aligned_count = (count / vec_width) * vec_width;

        // Vectorized mask checking and loading
        for (; i < aligned_count; i += vec_width) {
            // Load 8 masks (u8 → i32 for easier comparison)
            __m256i masks_lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(sample_mask + i)));

            // Check if any samples are active in this batch
            __m256i zero = _mm256_setzero_si256();
            __m256i cmp = _mm256_cmpeq_epi32(masks_lo, zero);
            int all_zero = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));

            if (all_zero == 0xFF) {
                // All samples masked out, skip
                continue;
            }

            // Process samples individually (histogram bins are scattered)
            // SIMD doesn't help much here due to random write pattern
            for (usize j = 0; j < vec_width; ++j) {
                if (sample_mask[i + j]) {
                    hist.add(features[i + j], gradients[i + j], hessians[i + j]);
                }
            }
        }

        // Handle remaining samples
        for (; i < count; ++i) {
            if (sample_mask[i]) {
                hist.add(features[i], gradients[i], hessians[i]);
            }
        }
    }
#endif

    //==========================================================================
    // NEON implementation (ARM)
    //==========================================================================

#if defined(SNAP_SIMD_NEON)
    static void build_neon(
        const i8* features,
        const i32* gradients,
        const i32* hessians,
        const u8* sample_mask,
        usize count,
        Histogram& hist
    ) {
        // Similar to AVX2 but using NEON intrinsics
        // Process 4 samples at a time

        usize i = 0;
        const usize vec_width = 4;
        const usize aligned_count = (count / vec_width) * vec_width;

        for (; i < aligned_count; i += vec_width) {
            // Load 4 masks
            uint8x8_t masks_u8 = vld1_u8(sample_mask + i);
            uint32x4_t masks_u32 = vmovl_u16(vget_low_u16(vmovl_u8(masks_u8)));

            // Check if any samples are active
            uint32x4_t zero = vdupq_n_u32(0);
            uint32x4_t cmp = vceqq_u32(masks_u32, zero);

            // Extract mask bits
            uint32_t mask_bits = vgetq_lane_u32(cmp, 0) & vgetq_lane_u32(cmp, 1) &
                                 vgetq_lane_u32(cmp, 2) & vgetq_lane_u32(cmp, 3);

            if (mask_bits == 0xFFFFFFFF) {
                // All samples masked out
                continue;
            }

            // Process individually
            for (usize j = 0; j < vec_width; ++j) {
                if (sample_mask[i + j]) {
                    hist.add(features[i + j], gradients[i + j], hessians[i + j]);
                }
            }
        }

        // Handle remaining samples
        for (; i < count; ++i) {
            if (sample_mask[i]) {
                hist.add(features[i], gradients[i], hessians[i]);
            }
        }
    }
#endif
};

} // namespace intgr
