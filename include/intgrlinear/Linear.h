#pragma once

#include "LinearModel.h"
#include "../intgrcore/Types.h"
#include "../intgrcore/FixedMath.h"

namespace intgr {

/// Linear model inference engine
/// Performs dot product and activation in fixed-point
class Linear {
public:
    /// Predict single sample (regression or raw logit for classification)
    /// @param model  Trained linear model
    /// @param features  Quantized feature vector (int8 array)
    /// @param num_features  Number of features (must match model.num_features)
    /// @return Prediction in fixed-point (int32 for regression, u8 for logistic)
    static i32 predict(const LinearModel& model, const i8* features, u16 num_features) noexcept;

    /// Predict with logistic activation (classification)
    /// @param model  Trained logistic model
    /// @param features  Quantized feature vector
    /// @param num_features  Number of features
    /// @return Probability scaled to [0, 255]
    static u8 predict_proba(const LinearModel& model, const i8* features, u16 num_features) noexcept;

    /// Predict batch of samples
    /// @param model  Trained model
    /// @param features  Feature matrix (row-major, int8)
    /// @param num_rows  Number of samples
    /// @param num_features  Number of features per sample
    /// @param out  Output array (int32, size = num_rows)
    static void predict_batch(const LinearModel& model,
                              const i8* features,
                              u32 num_rows,
                              u16 num_features,
                              i32* out) noexcept;

    /// Convert probability to class label (threshold at 0.5 = 128)
    /// @param proba  Probability in [0, 255]
    /// @return 0 or 1
    static i8 to_class(u8 proba) noexcept {
        return (proba >= 128) ? 1 : 0;
    }

private:
    /// Fixed-point dot product: y = w^T x + b
    /// @param weights  Weight vector (int16)
    /// @param features  Feature vector (int8)
    /// @param bias  Bias term (int16)
    /// @param num_features  Vector length
    /// @return Dot product in int32 (accumulated)
    static i32 dot_product(const i16* weights,
                          const i8* features,
                          i16 bias,
                          u16 num_features) noexcept;
};

} // namespace intgr
