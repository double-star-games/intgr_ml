#pragma once

#include "../intgrcore/Types.h"
#include "Precision.h"

namespace intgr {

/// Training parameters for gradient boosting
struct BoostParams {
    u32 num_trees;          ///< Number of boosting rounds
    u16 max_depth;          ///< Maximum tree depth
    u16 min_samples_leaf;   ///< Minimum samples per leaf node
    u8  learning_rate_q;    ///< Learning rate in fixed-point (FIXED_POINT_SCALE = 1.0)
    u8  task_type;          ///< 0=regression, 1=binary_classification
    u32 random_seed;        ///< Random seed for reproducibility
    i32 lambda;             ///< L2 regularization on leaf weights (default: 10)
    i64 min_gain;           ///< Minimum gain to make split (default: 1000)
    i32 gamma;              ///< Minimum sum of hessian in child (default: 100)
    PrecisionPolicy precision; ///< Mixed-precision configuration (default: int8-only)

    // Default constructor with sensible defaults tuned for int8 regime
    BoostParams()
        : num_trees(100)
        , max_depth(6)
        , min_samples_leaf(8)
        , learning_rate_q(32)  // 32/FIXED_POINT_SCALE ≈ 0.125
        , task_type(1)         // Binary classification
        , random_seed(42)
        , lambda(2)            // Light L2 regularization
        , min_gain(0)          // No minimum gain threshold (tune upward as needed)
        , gamma(1)             // Minimal hessian requirement
    {}

    /// Fluent API for parameter configuration
    BoostParams& trees(u32 n) noexcept {
        num_trees = n;
        return *this;
    }

    BoostParams& depth(u16 d) noexcept {
        max_depth = d;
        return *this;
    }

    BoostParams& min_leaf(u16 m) noexcept {
        min_samples_leaf = m;
        return *this;
    }

    BoostParams& learning_rate(u8 lr) noexcept {
        learning_rate_q = lr;
        return *this;
    }

    BoostParams& regression() noexcept {
        task_type = 0;
        return *this;
    }

    BoostParams& classification() noexcept {
        task_type = 1;
        return *this;
    }

    BoostParams& seed(u32 s) noexcept {
        random_seed = s;
        return *this;
    }

    /// Get learning rate as float (for display/logging)
    f32 learning_rate_float() const noexcept {
        return static_cast<f32>(learning_rate_q) / static_cast<f32>(constants::FIXED_POINT_SCALE);
    }
};

} // namespace intgr
