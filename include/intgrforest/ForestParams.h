#pragma once

#include "../intgrcore/Types.h"

namespace intgr {

/// Parameters for Random Forest training
struct ForestParams {
    u32 num_trees = 100;          ///< Number of trees in forest
    u32 max_depth = 6;            ///< Maximum tree depth
    u32 min_samples_split = 2;    ///< Minimum samples required to split node
    u32 min_samples_leaf = 1;     ///< Minimum samples required in leaf node
    u32 max_features = 0;         ///< Max features per tree (0 = sqrt(num_features))
    f32 sample_fraction = 1.0f;   ///< Bootstrap sample fraction (1.0 = 100% with replacement)
    u32 seed = 1337;              ///< Random seed for reproducibility
    u8  task_type = 0;            ///< 0=regression, 1=binary_classification
    bool parallel = true;         ///< Enable parallel tree building

    // Task type constants
    static constexpr u8 TASK_REGRESSION = 0;
    static constexpr u8 TASK_BINARY_CLASSIFICATION = 1;

    /// Fluent API: Set number of trees
    ForestParams& trees(u32 n) noexcept {
        num_trees = n;
        return *this;
    }

    /// Fluent API: Set maximum depth
    ForestParams& depth(u32 d) noexcept {
        max_depth = d;
        return *this;
    }

    /// Fluent API: Set minimum samples for split
    ForestParams& min_split(u32 m) noexcept {
        min_samples_split = m;
        return *this;
    }

    /// Fluent API: Set minimum samples in leaf
    ForestParams& min_leaf(u32 m) noexcept {
        min_samples_leaf = m;
        return *this;
    }

    /// Fluent API: Set max features per tree
    ForestParams& features(u32 f) noexcept {
        max_features = f;
        return *this;
    }

    /// Fluent API: Set bootstrap sample fraction
    ForestParams& bootstrap(f32 frac) noexcept {
        sample_fraction = frac;
        return *this;
    }

    /// Fluent API: Set random seed
    ForestParams& random_seed(u32 s) noexcept {
        seed = s;
        return *this;
    }

    /// Fluent API: Set task type (regression)
    ForestParams& regression() noexcept {
        task_type = TASK_REGRESSION;
        return *this;
    }

    /// Fluent API: Set task type (binary classification)
    ForestParams& classification() noexcept {
        task_type = TASK_BINARY_CLASSIFICATION;
        return *this;
    }

    /// Fluent API: Enable/disable parallel training
    ForestParams& use_parallel(bool enable) noexcept {
        parallel = enable;
        return *this;
    }
};

} // namespace intgr
