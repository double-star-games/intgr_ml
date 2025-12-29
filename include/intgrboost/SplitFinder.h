#pragma once

#include "Histogram.h"
#include "intgrcore/Types.h"
#include <limits>

namespace intgr {

/// Result of split search
struct SplitResult {
    i16 threshold;      ///< Best split threshold (int16 value, supports both int8 and int16 thresholds)
    i64 gain;           ///< Split gain (higher is better)
    i32 left_count;     ///< Number of samples in left child
    i32 right_count;    ///< Number of samples in right child
    bool found;         ///< True if valid split was found

    SplitResult()
        : threshold(0)
        , gain(std::numeric_limits<i64>::min())
        , left_count(0)
        , right_count(0)
        , found(false)
    {}
};

/// Find best split in a histogram using gradient-based gain
/// Uses XGBoost-style gain calculation: Gain = (G_L^2 / H_L) + (G_R^2 / H_R) - (G_total^2 / H_total)
///
/// @param hist             Histogram with gradient/hessian statistics
/// @param min_samples_leaf Minimum samples required in each child
/// @param lambda           L2 regularization parameter
/// @param min_gain         Minimum gain threshold to accept split
/// @param gamma            Minimum sum of hessian in each child
/// @return SplitResult with best threshold and gain, or found=false if no valid split
inline SplitResult find_best_split(
    const Histogram& hist,
    i32 min_samples_leaf,
    i32 lambda = 1,
    i64 min_gain = 0,
    i32 gamma = 0
) noexcept {
    SplitResult result;

    // Calculate totals
    i64 total_grad = hist.total_gradient();
    i64 total_hess = hist.total_hessian();
    i32 total_count = hist.total_count();

    // Edge case: no samples or invalid hessian
    if (total_count == 0 || total_hess <= 0) {
        return result;
    }

    // Edge case: not enough samples to split
    if (total_count < 2 * min_samples_leaf) {
        return result;
    }

    // Scan through possible split points
    // Split at bin i means: left gets bins [0, i], right gets bins [i+1, 255]
    i64 left_grad = 0;
    i64 left_hess = 0;
    i32 left_count = 0;

    i64 best_gain = std::numeric_limits<i64>::min();
    i16 best_threshold = 0;  // Changed to i16 to support both int8 and int16 thresholds
    i32 best_left_count = 0;

    // Try splits at each bin boundary (0 to 254)
    // We map bin index back to int8 threshold value
    for (usize bin = 0; bin < Histogram::NUM_BINS - 1; ++bin) {
        // Accumulate left statistics
        left_grad += hist.grad_sums[bin];
        left_hess += hist.hess_sums[bin];
        left_count += hist.counts[bin];

        // Calculate right statistics
        i64 right_grad = total_grad - left_grad;
        i64 right_hess = total_hess - left_hess;
        i32 right_count = total_count - left_count;

        // Check min_samples_leaf constraint
        if (left_count < min_samples_leaf || right_count < min_samples_leaf) {
            continue;
        }

        // Check for valid hessians (must be positive)
        if (left_hess <= 0 || right_hess <= 0) {
            continue;
        }

        // Check gamma constraint (minimum sum of hessian in children)
        if (gamma > 0 && (left_hess < gamma || right_hess < gamma)) {
            continue;
        }

        // Calculate gain using XGBoost formula:
        // Gain = (G_L^2 / (H_L + lambda)) + (G_R^2 / (H_R + lambda)) - (G_total^2 / (H_total + lambda))
        // Note: We omit the 0.5 factor since it doesn't affect which split is best

        i64 denom_left = left_hess + lambda;
        i64 denom_right = right_hess + lambda;
        i64 denom_total = total_hess + lambda;

        // Use integer arithmetic to avoid floating point
        // gain_left = left_grad^2 / denom_left
        i64 gain_left = (left_grad * left_grad) / denom_left;
        i64 gain_right = (right_grad * right_grad) / denom_right;
        i64 gain_parent = (total_grad * total_grad) / denom_total;

        i64 gain = gain_left + gain_right - gain_parent;

        // Check min_gain constraint
        if (gain < min_gain) {
            continue;
        }

        // Track best split
        if (gain > best_gain) {
            best_gain = gain;

            // Convert bin index to threshold value
            // bin 0 = value -128, bin 128 = value 0, bin 255 = value 127
            // We split such that values <= threshold go left
            // Note: Still using int8 range for now (256 bins), but storing as i16 for future extensibility
            best_threshold = static_cast<i16>(static_cast<i32>(bin) - constants::I8_OFFSET);
            best_left_count = left_count;
            result.found = true;
        }
    }

    if (result.found && best_gain >= min_gain) {
        result.threshold = best_threshold;
        result.gain = best_gain;
        result.left_count = best_left_count;
        result.right_count = total_count - best_left_count;
    } else {
        result.found = false;
    }

    return result;
}

} // namespace intgr
