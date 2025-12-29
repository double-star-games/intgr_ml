#pragma once

#include "ForestModel.h"
#include "../intgrcore/Types.h"
#include "../intgrboost/BoostModel.h"

namespace intgr {

/// Random Forest inference engine
/// Predicts by averaging predictions from all trees in the ensemble
class Forest {
public:
    /// Predict single sample (regression or raw score for classification)
    /// @param model  Trained forest model
    /// @param features  Quantized feature vector (int8 array)
    /// @param num_features  Number of features (must match model.num_features)
    /// @return Prediction in fixed-point (int32)
    static i32 predict(const ForestModel& model, const i8* features, u16 num_features) noexcept;

    /// Predict single tree within forest
    /// @param tree  Decision tree
    /// @param features  Quantized feature vector
    /// @return Tree prediction in fixed-point (int32)
    static i32 predict_tree(const Tree& tree, const i8* features) noexcept;

    /// Predict batch of samples
    /// @param model  Trained forest model
    /// @param features  Feature matrix (row-major, int8)
    /// @param num_rows  Number of samples
    /// @param num_features  Number of features per sample
    /// @param out  Output array (int32, size = num_rows)
    static void predict_batch(const ForestModel& model,
                              const i8* features,
                              u32 num_rows,
                              u16 num_features,
                              i32* out) noexcept;

    /// Convert raw prediction to class label (for classification)
    /// @param raw_pred  Raw prediction from predict()
    /// @return 0 or 1 for binary classification
    static i8 to_class(i32 raw_pred) noexcept {
        return (raw_pred >= 0) ? 1 : 0;
    }
};

} // namespace intgr
