#pragma once

#include "BoostModel.h"
#include "BoostParams.h"
#include "../intgrdata/Dataset.h"

namespace intgr {

/// Public API for IntgrBoost gradient boosting
class Boost {
public:
    /// Predict single sample (row-wise, int8 features)
    /// @param model  Trained BoostModel
    /// @param features  Feature vector (int8 quantized, length = model.num_features)
    /// @return Prediction score (fixed-point int32)
    static i32 predict_one(const BoostModel& model, const i8* features) noexcept;

    /// Batch predict on SoA dataset
    /// @param model  Trained BoostModel
    /// @param dataset  Quantized dataset (SoA layout)
    /// @param out_scores  Output buffer (length = dataset.rows())
    static void predict_batch(const BoostModel& model, const Dataset& dataset, i32* out_scores);

    /// Predict single tree
    /// @param tree  Single decision tree
    /// @param features  Feature vector (int8)
    /// @return Leaf value (fixed-point int32)
    static i32 predict_tree(const Tree& tree, const i8* features) noexcept;

private:
    // Prevent instantiation
    Boost() = delete;
};

} // namespace intgr
