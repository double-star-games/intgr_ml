#pragma once

#include "../intgrcore/Types.h"

namespace intgr {

/// Parameters for Linear model training
struct LinearParams {
    f32 learning_rate = 0.01f;   ///< Learning rate for gradient descent
    f32 l2_penalty = 0.0f;       ///< L2 regularization penalty
    u32 max_epochs = 100;        ///< Maximum training epochs
    u32 batch_size = 32;         ///< Mini-batch size (0 = full batch)
    f32 tolerance = 1e-4f;       ///< Convergence tolerance
    u32 seed = 1337;             ///< Random seed for mini-batch sampling
    u8  task_type = 0;           ///< 0=regression, 1=logistic_classification
    bool verbose = false;        ///< Print training progress

    // Task type constants
    static constexpr u8 TASK_REGRESSION = 0;
    static constexpr u8 TASK_LOGISTIC = 1;

    /// Fluent API: Set learning rate
    LinearParams& lr(f32 rate) noexcept {
        learning_rate = rate;
        return *this;
    }

    /// Fluent API: Set L2 penalty
    LinearParams& l2(f32 penalty) noexcept {
        l2_penalty = penalty;
        return *this;
    }

    /// Fluent API: Set max epochs
    LinearParams& epochs(u32 n) noexcept {
        max_epochs = n;
        return *this;
    }

    /// Fluent API: Set batch size
    LinearParams& batch(u32 size) noexcept {
        batch_size = size;
        return *this;
    }

    /// Fluent API: Set convergence tolerance
    LinearParams& tol(f32 t) noexcept {
        tolerance = t;
        return *this;
    }

    /// Fluent API: Set random seed
    LinearParams& random_seed(u32 s) noexcept {
        seed = s;
        return *this;
    }

    /// Fluent API: Set task type (regression)
    LinearParams& regression() noexcept {
        task_type = TASK_REGRESSION;
        return *this;
    }

    /// Fluent API: Set task type (logistic classification)
    LinearParams& logistic() noexcept {
        task_type = TASK_LOGISTIC;
        return *this;
    }

    /// Fluent API: Enable verbose output
    LinearParams& print_progress(bool enable = true) noexcept {
        verbose = enable;
        return *this;
    }
};

} // namespace intgr
