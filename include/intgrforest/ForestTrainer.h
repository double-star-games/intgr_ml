#pragma once

#include "ForestModel.h"
#include "ForestParams.h"
#include "../intgrdata/Dataset.h"
#include "../intgrboost/TreeBuilder.h"
#include "../intgrcore/Types.h"
#include <vector>
#include <random>

namespace intgr {

/// Random Forest trainer with bagging and feature randomization
class ForestTrainer {
public:
    /// Train random forest on quantized dataset
    /// @param dataset  Quantized training data (int8 features)
    /// @param labels   Target labels (fixed-point int32)
    /// @param params   Training parameters
    /// @return Trained forest model
    static ForestModel train(const Dataset& dataset,
                            const i32* labels,
                            const ForestParams& params);
    /// Bootstrap sample: sample with replacement
    /// @param num_samples  Original dataset size
    /// @param sample_fraction  Fraction to sample (1.0 = 100%)
    /// @param rng  Random number generator
    /// @return Indices of sampled rows
    static std::vector<u32> bootstrap_sample(u32 num_samples,
                                             f32 sample_fraction,
                                             std::mt19937& rng);

    /// Random feature selection for one tree
    /// @param num_features  Total number of features
    /// @param max_features  Maximum features to select (0 = sqrt rule)
    /// @param rng  Random number generator
    /// @return Indices of selected features
    static std::vector<u16> select_features(u16 num_features,
                                           u32 max_features,
                                           std::mt19937& rng);

    /// Build single tree on bootstrap sample
    /// @param dataset  Full dataset
    /// @param labels  All labels
    /// @param sample_indices  Bootstrap sample indices
    /// @param feature_indices  Selected feature indices
    /// @param params  Training parameters
    /// @param is_binary_classification  Whether this is binary (2-class) classification
    /// @return Trained tree
    static Tree build_tree(const Dataset& dataset,
                          const i32* labels,
                          const std::vector<u32>& sample_indices,
                          const std::vector<u16>& feature_indices,
                          const ForestParams& params,
                          bool is_binary_classification);
};

} // namespace intgr
