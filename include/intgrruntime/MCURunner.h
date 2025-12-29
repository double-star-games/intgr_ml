#pragma once

#include "../intgrcore/Types.h"
#include "../intgrboost/BoostModel.h"

namespace intgr {
namespace runtime {

/// MCU Device Runner - Integer-only model evaluator
/// Simulates on-device inference with zero floating-point operations
/// Designed for embedded systems with limited compute resources
class MCURunner {
public:
    /// Load MCU-format model from file
    /// @param path  Path to .mcu model file
    /// @return True on success, false on error
    bool load(const char* path);

    /// Predict single sample (batch size = 1)
    /// @param features  Quantized feature vector (int8 array, length = num_features)
    /// @return Prediction logit (int32 fixed-point)
    i32 predict_one(const i8* features) const noexcept;

    /// Predict batch of samples
    /// @param features  Feature matrix (row-major, int8)
    /// @param num_rows  Number of samples
    /// @param out_logits  Output array (int32, size = num_rows or num_rows * num_classes)
    void predict_batch(const i8* features, u32 num_rows, i32* out_logits) const noexcept;

    /// Get model metadata
    u32 num_trees() const noexcept { return num_trees_; }
    u16 num_features() const noexcept { return num_features_; }
    u32 num_classes() const noexcept { return num_classes_; }
    bool is_prescaled() const noexcept { return prescaled_; }
    u16 model_type() const noexcept { return model_type_; }

private:
    /// Predict single tree
    i32 predict_tree(const Tree& tree, const i8* features) const noexcept;

    /// Model metadata
    u16 model_type_;     // 0=boost, 1=forest, 2=ovr
    u32 num_trees_;
    u16 num_features_;
    u32 num_classes_;
    bool prescaled_;
    i32 learning_rate_fp_;  // Q8.8 learning rate (stored for non-prescaled models)
    i32 bias_fp_;            // Global bias (Q0 for boost, 0 for forest)

    /// Model trees
    /// For boost/forest: single vector of trees
    /// For OvR: concatenated trees (num_classes * trees_per_head)
    std::vector<Tree> trees_;
    u32 trees_per_head_;  // For OvR models only
};

} // namespace runtime
} // namespace intgr
