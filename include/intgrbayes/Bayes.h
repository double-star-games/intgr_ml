#pragma once

#include "BayesModel.h"
#include "../intgrcore/Types.h"

namespace intgr {

/// Naive Bayes inference engine
/// Uses log-domain arithmetic for numerical stability
class Bayes {
public:
    /// Predict class for a single sample
    /// @param model     Trained Bayes model
    /// @param features  Input features (int8, length = num_features)
    /// @return Predicted class index
    static u16 predict(const BayesModel& model, const i8* features) noexcept;

    /// Compute log probabilities for all classes
    /// @param model     Trained Bayes model
    /// @param features  Input features
    /// @param output    Output log probs (int32, length = num_classes)
    static void predict_log_proba(const BayesModel& model,
                                  const i8* features,
                                  i32* output) noexcept;

    /// Predict class for multiple samples (batch)
    /// @param model       Trained Bayes model
    /// @param features    Input matrix (row-major, num_samples × num_features)
    /// @param num_samples Number of samples
    /// @param output      Output class indices (length = num_samples)
    static void predict_batch(const BayesModel& model,
                             const i8* features,
                             u32 num_samples,
                             u16* output) noexcept;

private:
    /// Compute log P(class | features) for a single class
    /// Returns log probability as int32 (sum of int16 log probs)
    static i32 compute_class_log_prob(const BayesModel& model,
                                     const i8* features,
                                     u16 class_idx) noexcept;
};

} // namespace intgr
