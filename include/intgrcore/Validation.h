#pragma once

#include "intgrcore/Result.h"
#include "intgrcore/Types.h"
#include <string>

namespace intgr {

/// Validation results for training parameters
/// These are pure functions with no side effects - fully testable

/// Validate number of trees for boosting/forest
/// Valid range: 1 to 10000
inline Result<void> validate_num_trees(u32 num_trees) {
    if (num_trees == 0) {
        return Err(ErrorCode::OutOfRange, "Number of trees must be at least 1");
    }
    if (num_trees > 10000) {
        return Err(ErrorCode::OutOfRange, "Number of trees must not exceed 10000");
    }
    return Ok();
}

/// Validate max depth for tree-based models
/// Valid range: 1 to 30 (forest allows deeper trees than boost)
inline Result<void> validate_max_depth(u32 max_depth, u32 max_allowed = 30) {
    if (max_depth == 0) {
        return Err(ErrorCode::OutOfRange, "Max depth must be at least 1");
    }
    if (max_depth > max_allowed) {
        return Err(ErrorCode::OutOfRange, "Max depth must not exceed " + std::to_string(max_allowed));
    }
    return Ok();
}

/// Validate learning rate for boosting
/// Valid range: (0.0, 1.0]
inline Result<void> validate_learning_rate(float lr) {
    if (lr <= 0.0f) {
        return Err(ErrorCode::OutOfRange, "Learning rate must be greater than 0");
    }
    if (lr > 1.0f) {
        return Err(ErrorCode::OutOfRange, "Learning rate must not exceed 1.0");
    }
    return Ok();
}

/// Validate min samples per leaf
/// Valid range: 1 to 1000
inline Result<void> validate_min_samples_leaf(u16 min_samples) {
    if (min_samples == 0) {
        return Err(ErrorCode::OutOfRange, "Min samples per leaf must be at least 1");
    }
    if (min_samples > 1000) {
        return Err(ErrorCode::OutOfRange, "Min samples per leaf must not exceed 1000");
    }
    return Ok();
}

/// Validate clip range for quantization
/// Requirements: min < max, reasonable range
inline Result<void> validate_clip_range(float clip_min, float clip_max) {
    if (clip_min >= clip_max) {
        return Err(ErrorCode::InvalidArgument, "Clip min must be less than clip max");
    }
    if (clip_max - clip_min < 0.01f) {
        return Err(ErrorCode::InvalidArgument, "Clip range must be at least 0.01");
    }
    return Ok();
}

/// Validate quantization precision
/// Valid range: (0.0, 1.0]
inline Result<void> validate_precision(float precision) {
    if (precision <= 0.0f) {
        return Err(ErrorCode::OutOfRange, "Precision must be greater than 0");
    }
    if (precision > 1.0f) {
        return Err(ErrorCode::OutOfRange, "Precision must not exceed 1.0");
    }
    return Ok();
}

/// Validate number of classes for multiclass
/// Valid range: 2 to 1000
inline Result<void> validate_num_classes(u32 num_classes) {
    if (num_classes < 2) {
        return Err(ErrorCode::OutOfRange, "Number of classes must be at least 2");
    }
    if (num_classes > 1000) {
        return Err(ErrorCode::OutOfRange, "Number of classes must not exceed 1000");
    }
    return Ok();
}

/// Validate number of PCA components
/// Valid range: 1 to num_features
inline Result<void> validate_num_components(u32 num_components, u32 num_features) {
    if (num_components == 0) {
        return Err(ErrorCode::OutOfRange, "Number of components must be at least 1");
    }
    if (num_components > num_features) {
        return Err(ErrorCode::OutOfRange,
            "Number of components (" + std::to_string(num_components) +
            ") cannot exceed number of features (" + std::to_string(num_features) + ")");
    }
    return Ok();
}

/// Validate batch size for training
/// Valid range: 1 to 100000
inline Result<void> validate_batch_size(u32 batch_size) {
    if (batch_size == 0) {
        return Err(ErrorCode::OutOfRange, "Batch size must be at least 1");
    }
    if (batch_size > 100000) {
        return Err(ErrorCode::OutOfRange, "Batch size must not exceed 100000");
    }
    return Ok();
}

/// Validate number of epochs
/// Valid range: 1 to 10000
inline Result<void> validate_epochs(u32 epochs) {
    if (epochs == 0) {
        return Err(ErrorCode::OutOfRange, "Number of epochs must be at least 1");
    }
    if (epochs > 10000) {
        return Err(ErrorCode::OutOfRange, "Number of epochs must not exceed 10000");
    }
    return Ok();
}

/// Validate L2 regularization penalty
/// Valid range: [0.0, 1.0]
inline Result<void> validate_l2_penalty(float l2) {
    if (l2 < 0.0f) {
        return Err(ErrorCode::OutOfRange, "L2 penalty cannot be negative");
    }
    if (l2 > 1.0f) {
        return Err(ErrorCode::OutOfRange, "L2 penalty must not exceed 1.0");
    }
    return Ok();
}

/// Validate sample fraction for bagging
/// Valid range: (0.0, 1.0]
inline Result<void> validate_sample_fraction(float fraction) {
    if (fraction <= 0.0f) {
        return Err(ErrorCode::OutOfRange, "Sample fraction must be greater than 0");
    }
    if (fraction > 1.0f) {
        return Err(ErrorCode::OutOfRange, "Sample fraction must not exceed 1.0");
    }
    return Ok();
}

/// Validate feature fraction for random subspace
/// Valid range: (0.0, 1.0]
inline Result<void> validate_feature_fraction(float fraction) {
    if (fraction <= 0.0f) {
        return Err(ErrorCode::OutOfRange, "Feature fraction must be greater than 0");
    }
    if (fraction > 1.0f) {
        return Err(ErrorCode::OutOfRange, "Feature fraction must not exceed 1.0");
    }
    return Ok();
}

/// Combined validation for BoostParams
struct BoostParamsValidation {
    u32 num_trees = 100;
    u32 max_depth = 6;
    u16 min_samples_leaf = 8;
    float learning_rate = 0.125f;

    static constexpr u32 MAX_BOOST_DEPTH = 20;

    Result<void> validate() const {
        auto r = validate_num_trees(num_trees);
        if (!r.ok()) return r;

        r = validate_max_depth(max_depth, MAX_BOOST_DEPTH);
        if (!r.ok()) return r;

        r = validate_min_samples_leaf(min_samples_leaf);
        if (!r.ok()) return r;

        r = validate_learning_rate(learning_rate);
        if (!r.ok()) return r;

        return Ok();
    }
};

/// Combined validation for ForestParams
struct ForestParamsValidation {
    u32 num_trees = 100;
    u32 max_depth = 10;
    float sample_fraction = 0.8f;
    float feature_fraction = 0.7f;

    static constexpr u32 MAX_FOREST_DEPTH = 30;

    Result<void> validate() const {
        auto r = validate_num_trees(num_trees);
        if (!r.ok()) return r;

        r = validate_max_depth(max_depth, MAX_FOREST_DEPTH);
        if (!r.ok()) return r;

        r = validate_sample_fraction(sample_fraction);
        if (!r.ok()) return r;

        r = validate_feature_fraction(feature_fraction);
        if (!r.ok()) return r;

        return Ok();
    }
};

/// Combined validation for LinearParams
struct LinearParamsValidation {
    float learning_rate = 0.01f;
    float l2_penalty = 0.001f;
    u32 epochs = 100;
    u32 batch_size = 128;

    Result<void> validate() const {
        auto r = validate_learning_rate(learning_rate);
        if (!r.ok()) return r;

        r = validate_l2_penalty(l2_penalty);
        if (!r.ok()) return r;

        r = validate_epochs(epochs);
        if (!r.ok()) return r;

        r = validate_batch_size(batch_size);
        if (!r.ok()) return r;

        return Ok();
    }
};

} // namespace intgr
