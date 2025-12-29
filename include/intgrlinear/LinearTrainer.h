#pragma once

#include "LinearModel.h"
#include "LinearParams.h"
#include "../intgrdata/Dataset.h"
#include "../intgrcore/Types.h"

namespace intgr {

/// Linear model trainer using fixed-point gradient descent
class LinearTrainer {
public:
    /// Train linear model on quantized dataset
    /// @param dataset  Quantized training data (int8 features)
    /// @param labels   Target labels (fixed-point int32)
    /// @param params   Training parameters
    /// @return Trained linear model
    static LinearModel train(const Dataset& dataset,
                            const i32* labels,
                            const LinearParams& params);

private:
    /// Compute gradient for regression (MSE loss)
    /// @param model  Current model
    /// @param features  Feature matrix
    /// @param labels  Target values
    /// @param num_samples  Number of samples
    /// @param num_features  Number of features
    /// @param grad_weights  Output: gradient w.r.t. weights
    /// @param grad_bias  Output: gradient w.r.t. bias
    static void compute_gradient_regression(
        const LinearModel& model,
        const i8* features,
        const i32* labels,
        usize num_samples,
        u16 num_features,
        i32* grad_weights,
        i32* grad_bias);

    /// Compute gradient for logistic classification (BCE loss)
    /// @param model  Current model
    /// @param features  Feature matrix
    /// @param labels  Target labels (0 or 1)
    /// @param num_samples  Number of samples
    /// @param num_features  Number of features
    /// @param grad_weights  Output: gradient w.r.t. weights
    /// @param grad_bias  Output: gradient w.r.t. bias
    static void compute_gradient_logistic(
        const LinearModel& model,
        const i8* features,
        const i32* labels,
        usize num_samples,
        u16 num_features,
        i32* grad_weights,
        i32* grad_bias);

    /// Update weights with gradient descent
    /// @param model  Model to update
    /// @param grad_weights  Weight gradients
    /// @param grad_bias  Bias gradient
    /// @param learning_rate  Learning rate
    /// @param l2_penalty  L2 regularization
    static void update_weights(
        LinearModel& model,
        const i32* grad_weights,
        i32 grad_bias,
        f32 learning_rate,
        f32 l2_penalty);

    /// Compute loss for convergence check
    /// @param model  Current model
    /// @param features  Feature matrix
    /// @param labels  Target values
    /// @param num_samples  Number of samples
    /// @param num_features  Number of features
    /// @return Loss value
    static f32 compute_loss(
        const LinearModel& model,
        const i8* features,
        const i32* labels,
        usize num_samples,
        u16 num_features);
};

} // namespace intgr
