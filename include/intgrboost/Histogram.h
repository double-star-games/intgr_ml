#pragma once

#include "intgrcore/Types.h"
#include <cstring>

namespace intgr {

/// Histogram for gradient boosting split finding
/// Uses 256 bins for int8 quantized features
struct Histogram {
    static constexpr usize NUM_BINS = constants::HISTOGRAM_BINS;

    i32 counts[NUM_BINS];      // Number of samples in each bin
    i64 grad_sums[NUM_BINS];   // Sum of gradients in each bin (int64 to avoid overflow)
    i64 hess_sums[NUM_BINS];   // Sum of hessians in each bin

    /// Clear histogram (set all bins to zero)
    void clear() noexcept {
        std::memset(counts, 0, sizeof(counts));
        std::memset(grad_sums, 0, sizeof(grad_sums));
        std::memset(hess_sums, 0, sizeof(hess_sums));
    }

    /// Add a single sample to the histogram
    /// @param feature_val  Quantized feature value (int8: -128 to 127)
    /// @param gradient     Gradient value (int32)
    /// @param hessian      Hessian value (int32)
    void add(i8 feature_val, i32 gradient, i32 hessian) noexcept {
        // Map int8 [-128, 127] to bin index [0, 255] in numeric order
        // bin 0 = value -128, bin 128 = value 0, bin 255 = value 127
        usize bin_idx = static_cast<usize>(static_cast<i32>(feature_val) + constants::I8_OFFSET);

        counts[bin_idx]++;
        grad_sums[bin_idx] += gradient;
        hess_sums[bin_idx] += hessian;
    }

    /// Get total count across all bins (property test)
    i32 total_count() const noexcept {
        i64 sum = 0;
        for (usize i = 0; i < NUM_BINS; ++i) {
            sum += counts[i];
        }
        return static_cast<i32>(sum);
    }

    /// Get total gradient sum across all bins (property test)
    i64 total_gradient() const noexcept {
        i64 sum = 0;
        for (usize i = 0; i < NUM_BINS; ++i) {
            sum += grad_sums[i];
        }
        return sum;
    }

    /// Get total hessian sum across all bins (property test)
    i64 total_hessian() const noexcept {
        i64 sum = 0;
        for (usize i = 0; i < NUM_BINS; ++i) {
            sum += hess_sums[i];
        }
        return sum;
    }
};

/// Build histogram for a single feature
/// @param hist          Output histogram (will be cleared first)
/// @param feature_vals  Quantized feature values (int8 array)
/// @param gradients     Gradient values (int32 array)
/// @param hessians      Hessian values (int32 array)
/// @param num_samples   Number of samples
inline void build_histogram(
    Histogram& hist,
    const i8* feature_vals,
    const i32* gradients,
    const i32* hessians,
    usize num_samples
) noexcept {
    hist.clear();

    for (usize i = 0; i < num_samples; ++i) {
        hist.add(feature_vals[i], gradients[i], hessians[i]);
    }
}

} // namespace intgr
