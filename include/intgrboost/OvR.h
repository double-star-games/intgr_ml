#pragma once

#include "OvRModel.h"
#include "Boost.h"
#include "../intgrdata/Dataset.h"
#include <algorithm>

namespace intgr {

/// Public API for OvR (One-vs-Rest) multiclass classification
class OvR {
public:
    /// Predict single sample - returns class index (0..K-1)
    /// @param model  Trained OvRModel
    /// @param features  Feature vector (int8 quantized, length = model.num_features)
    /// @return Predicted class index (argmax of K head scores)
    static i32 predict_one(const OvRModel& model, const i8* features) noexcept {
        if (model.heads.empty()) {
            return 0;
        }

        // Get logits from all K heads
        i32 max_score = std::numeric_limits<i32>::min();
        i32 best_class = 0;

        for (u32 k = 0; k < model.num_classes; ++k) {
            i32 score = Boost::predict_one(model.heads[k], features);

            if (score > max_score) {
                max_score = score;
                best_class = static_cast<i32>(k);
            }
        }

        return best_class;
    }

    /// Get logits for all K classes (for debugging or probability estimation)
    /// @param model  Trained OvRModel
    /// @param features  Feature vector (int8 quantized)
    /// @param out_logits  Output buffer (size = model.num_classes)
    static void predict_logits(const OvRModel& model, const i8* features, i32* out_logits) noexcept {
        for (u32 k = 0; k < model.num_classes; ++k) {
            out_logits[k] = Boost::predict_one(model.heads[k], features);
        }
    }

    /// Batch predict on SoA dataset - returns class indices
    /// @param model  Trained OvRModel
    /// @param dataset  Quantized dataset (SoA layout)
    /// @param out_classes  Output buffer for class predictions (length = dataset.rows())
    static void predict_batch(const OvRModel& model, const Dataset& dataset, i32* out_classes) {
        const usize num_samples = dataset.rows();
        const usize num_features = dataset.cols();

        // Hoist allocation outside loop (O(1) instead of O(n))
        std::vector<i8> row_features(num_features);

        // Row-by-row prediction (future: optimize with batch processing)
        for (usize row = 0; row < num_samples; ++row) {
            // Extract row into reused buffer
            dataset.copy_row_i8(row, row_features.data());

            // Predict
            out_classes[row] = predict_one(model, row_features.data());
        }
    }

    /// Batch predict logits for all classes
    /// @param model  Trained OvRModel
    /// @param dataset  Quantized dataset
    /// @param out_logits  Output buffer (size = dataset.rows() * model.num_classes)
    ///                    Layout: [sample0_class0, sample0_class1, ..., sample1_class0, ...]
    static void predict_batch_logits(const OvRModel& model, const Dataset& dataset, i32* out_logits) {
        const usize num_samples = dataset.rows();
        const usize num_features = dataset.cols();

        // Hoist allocation outside loop (O(1) instead of O(n))
        std::vector<i8> row_features(num_features);

        for (usize row = 0; row < num_samples; ++row) {
            // Extract row into reused buffer
            dataset.copy_row_i8(row, row_features.data());

            // Predict all K logits for this sample
            predict_logits(model, row_features.data(), &out_logits[row * model.num_classes]);
        }
    }

private:
    // Prevent instantiation
    OvR() = delete;
};

} // namespace intgr
