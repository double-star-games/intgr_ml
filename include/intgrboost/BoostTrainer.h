#pragma once

#include "BoostModel.h"
#include "BoostParams.h"
#include "TreeBuilder.h"
#include "Boost.h"
#include "../intgrcore/Limits.h"
#include "../intgrdata/Dataset.h"
#include <vector>
#include <algorithm>
#include <set>

namespace intgr {

/// Gradient boosting trainer
class BoostTrainer {
public:
    /// Train a gradient boosting model
    /// @param dataset  Training dataset (quantized)
    /// @param params   Training parameters
    /// @return Trained BoostModel
    static BoostModel train(const Dataset& dataset, const BoostParams& params) {
        usize num_samples = dataset.rows();

        if (num_samples == 0) {
            throw std::runtime_error("Cannot train on empty dataset");
        }

        BoostModel model;

        // Validate feature count against intgr_ml v1 limit
        check_feature_limit(dataset.cols());
        model.num_features = static_cast<u16>(dataset.cols());
        model.learning_rate_fp = params.learning_rate_q;

        // Q16.16 fixed-point format provides 16 bits fractional precision, eliminating quantization loss
        std::vector<i64> predictions_q16(num_samples, 0);

        // Calculate initial bias
        if (params.task_type == 1) {  // Classification
            // Check if multiclass by counting unique labels
            std::set<i32> unique_labels;
            const i32* labels = dataset.labels();
            for (usize i = 0; i < num_samples && unique_labels.size() <= 2; ++i) {
                unique_labels.insert(labels[i]);
            }

            if (unique_labels.size() <= 2) {
                // Binary classification: use log-odds
                i64 positive_count = 0;
                for (usize i = 0; i < num_samples; ++i) {
                    if (labels[i] == 1) {
                        positive_count++;
                    }
                }
                // Start at 0 (equivalent to 50% probability)
                model.bias_fp = 0;
            } else {
                // Multiclass: treat as regression, bias = mean of labels (scaled by 32)
                // Scaling factor of 32 puts labels in comparable range to hessian=256
                // Note: scale BEFORE division to avoid integer truncation (e.g., 119/120=0)
                i64 sum_labels = 0;
                for (usize i = 0; i < num_samples; ++i) {
                    sum_labels += labels[i];
                }
                model.bias_fp = static_cast<i32>((sum_labels * 32) / static_cast<i64>(num_samples));
            }
        } else {
            // Regression: bias = mean of labels
            i64 sum_labels = 0;
            const i32* labels = dataset.labels();
            for (usize i = 0; i < num_samples; ++i) {
                sum_labels += labels[i];
            }
            model.bias_fp = static_cast<i32>(sum_labels / static_cast<i64>(num_samples));
        }

        i64 bias_q16 = static_cast<i64>(model.bias_fp) << 16;
        for (usize i = 0; i < num_samples; ++i) {
            predictions_q16[i] = bias_q16;
        }

        std::vector<i32> gradients(num_samples);
        std::vector<i32> hessians(num_samples);
        std::vector<u8> sample_mask(num_samples, 1);  // All samples active

        // Detect if multiclass (>2 unique labels)
        std::set<i32> unique_labels_set;
        const i32* labels = dataset.labels();
        for (usize i = 0; i < num_samples && unique_labels_set.size() <= 2; ++i) {
            unique_labels_set.insert(labels[i]);
        }
        bool is_multiclass = (params.task_type == 1 && unique_labels_set.size() > 2);

        TreeBuilder builder(params);

        for (u32 iter = 0; iter < params.num_trees; ++iter) {
            // Convert Q16.16 predictions to Q0 for gradient computation
            std::vector<i64> predictions_q0(num_samples);
            for (usize i = 0; i < num_samples; ++i) {
                predictions_q0[i] = predictions_q16[i] >> 16;
            }

            compute_gradients_hessians(
                dataset,
                predictions_q0.data(),
                params.task_type,
                is_multiclass,
                gradients.data(),
                hessians.data()
            );

            Tree tree = builder.build(
                dataset,
                gradients.data(),
                hessians.data(),
                sample_mask.data()
            );

            // Update predictions with new tree in Q16.16 (NO division = NO precision loss!)
            // tree_pred is Q0, lr_q is Q8.8, so (tree_pred * lr_q) << 8 gives Q16.16
            i64 lr_scale_q16 = static_cast<i64>(params.learning_rate_q) << 8;  // Convert Q8.8 to Q16.16 scale

            for (usize i = 0; i < num_samples; ++i) {
                std::vector<i8> features(dataset.cols());
                dataset.copy_row_i8(i, features.data());

                i32 tree_pred = Boost::predict_tree(tree, features.data());

                // Accumulate in Q16.16: tree_pred (Q0) * lr_scale_q16 (Q16.16) → Q16.16, no division = perfect precision
                predictions_q16[i] += static_cast<i64>(tree_pred) * lr_scale_q16;
            }

            model.add_tree(std::move(tree));
        }

        return model;
    }

private:
    /// Fast integer sigmoid approximation using piecewise linear interpolation
    /// Maps fixed-point prediction score → sigmoid value scaled to [0, 256]
    /// Uses 5-segment piecewise linear approximation for sigmoid curve
    static i32 sigmoid_int(i64 x) noexcept {
        // Clamp to prevent overflow (sigmoid saturates outside this range)
        if (x <= -2048) return 0;    // sigmoid(-8) ≈ 0.0003
        if (x >= 2048) return 256;   // sigmoid(+8) ≈ 0.9997

        // Piecewise linear approximation (5 segments)
        // sigmoid(x) ≈ 0.5 + x/4 near x=0, but steeper at tails

        if (x >= 1024) {
            // [4, 8]: sigmoid(x) ≈ 0.98 + (x-4)*0.019
            // Map [1024, 2048] → [251, 256]
            return 251 + static_cast<i32>((x - 1024) * 5 / 1024);
        } else if (x >= 256) {
            // [1, 4]: sigmoid(x) ≈ 0.73 + (x-1)*0.083
            // Map [256, 1024] → [187, 251]
            return 187 + static_cast<i32>((x - 256) * 64 / 768);
        } else if (x >= -256) {
            // [-1, 1]: sigmoid(x) ≈ 0.5 + x*0.25
            // Map [-256, 256] → [64, 192]
            return 128 + static_cast<i32>(x / 2);
        } else if (x >= -1024) {
            // [-4, -1]: sigmoid(x) ≈ 0.27 - (x+1)*0.083
            // Map [-1024, -256] → [5, 69]
            return 69 - static_cast<i32>((-x - 256) * 64 / 768);
        } else {
            // [-8, -4]: sigmoid(x) ≈ 0.02 + (x+4)*0.019
            // Map [-2048, -1024] → [0, 5]
            return 5 - static_cast<i32>((-x - 1024) * 5 / 1024);
        }
    }

    /// Compute gradients and hessians for current predictions
    static void compute_gradients_hessians(
        const Dataset& dataset,
        const i64* predictions,
        u8 task_type,
        bool is_multiclass,
        i32* gradients,
        i32* hessians
    ) {
        usize num_samples = dataset.rows();
        const i32* labels = dataset.labels();

        if (task_type == 1 && !is_multiclass) {
            // Binary classification with logistic loss
            // gradient = sigmoid(pred) - label
            // hessian = sigmoid(pred) * (1 - sigmoid(pred))
            // Scale by 256 for integer precision

            for (usize i = 0; i < num_samples; ++i) {
                i64 pred = predictions[i];

                // Compute sigmoid with improved approximation
                i32 sig_scaled = sigmoid_int(pred);  // Returns value in [0, 256]

                // gradient = (sigmoid - label) scaled for precision
                // sig_scaled is already in [0, 256] representing [0, 1]
                // So gradient is in [-256, 256] representing [-1, 1]
                i32 target_scaled = labels[i] * 256;
                gradients[i] = (sig_scaled - target_scaled);

                // hessian = sigmoid * (1 - sigmoid)
                // sig_scaled in [0, 256], so hessian in [0, 64] representing [0, 0.25]
                i32 hess = (sig_scaled * (256 - sig_scaled)) / 256;
                hessians[i] = (hess > 0) ? hess : 1;
            }
        } else if (task_type == 1 && is_multiclass) {
            // Multiclass classification (treated as regression with scaled labels)
            // gradient = pred - label*32 (scaled to match hessian magnitude)
            // hessian = 256 (constant, scaled)

            for (usize i = 0; i < num_samples; ++i) {
                gradients[i] = (static_cast<i32>(predictions[i]) - labels[i] * 32);
                hessians[i] = 256;  // Keep constant for stability
            }
        } else {
            // Pure regression with squared loss
            // gradient = pred - label (unscaled)
            // hessian = 256 (constant, scaled)

            for (usize i = 0; i < num_samples; ++i) {
                gradients[i] = (static_cast<i32>(predictions[i]) - labels[i]);
                hessians[i] = 256;  // Keep constant for stability
            }
        }
    }
};

} // namespace intgr
