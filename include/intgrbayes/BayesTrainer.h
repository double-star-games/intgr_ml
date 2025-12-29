#pragma once

#include "BayesModel.h"
#include "BayesParams.h"
#include "../intgrdata/Dataset.h"
#include <vector>

namespace intgr {

/// Naive Bayes trainer
/// Computes class priors and feature likelihoods from training data
class BayesTrainer {
public:
    /// Train Naive Bayes model from dataset
    /// @param dataset  Training data
    /// @param labels   Class labels (0, 1, 2, ...)
    /// @param params   Training parameters
    /// @return Trained Bayes model
    static BayesModel train(const Dataset& dataset,
                           const i32* labels,
                           const BayesParams& params);

private:
    /// Count samples per class
    static std::vector<u32> count_classes(const i32* labels,
                                         usize num_samples,
                                         u16 num_classes);

    /// Count feature occurrences: count[class][feature][value]
    /// Returns flattened 3D array
    static std::vector<u32> count_features(const Dataset& dataset,
                                          const i32* labels,
                                          u16 num_classes,
                                          u16 num_bins);

    /// Compute class log priors with Laplace smoothing
    static std::vector<i16> compute_class_priors(const std::vector<u32>& class_counts,
                                                 usize num_samples,
                                                 f32 alpha,
                                                 i32 scale_factor);

    /// Compute feature log probabilities with Laplace smoothing
    static std::vector<i16> compute_feature_probs(const std::vector<u32>& feature_counts,
                                                  const std::vector<u32>& class_counts,
                                                  u16 num_classes,
                                                  u16 num_features,
                                                  u16 num_bins,
                                                  f32 alpha,
                                                  i32 scale_factor);

    /// Convert probability to scaled log probability (int16)
    static i16 to_log_prob(f32 prob, i32 scale_factor) noexcept;

    /// Find number of unique classes in labels
    static u16 find_num_classes(const i32* labels, usize num_samples);
};

} // namespace intgr
