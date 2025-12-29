#pragma once

#include "OvRModel.h"
#include "BoostParams.h"
#include "BoostTrainer.h"
#include "../intgrcore/Limits.h"
#include "../intgrdata/Dataset.h"
#include <vector>
#include <algorithm>
#include <set>
#include <future>
#include <thread>

namespace intgr {

/// One-vs-Rest multiclass gradient boosting trainer
class OvRTrainer {
public:
    /// Train a multiclass OvR model using K binary IntgrBoost classifiers
    /// @param dataset  Training dataset (quantized)
    /// @param params   Training parameters (will be used for each binary head)
    /// @param parallel Whether to train heads in parallel (default: true)
    /// @return Trained OvRModel
    static OvRModel train(const Dataset& dataset, const BoostParams& params, bool parallel = true) {
        // Validate feature count against intgr_ml v1 limit
        check_feature_limit(dataset.cols());

        usize num_samples = dataset.rows();

        if (num_samples == 0) {
            throw std::runtime_error("Cannot train on empty dataset");
        }

        // Detect number of classes from labels
        const i32* labels = dataset.labels();
        std::set<i32> unique_labels;
        i32 min_label = labels[0];
        i32 max_label = labels[0];

        for (usize i = 0; i < num_samples; ++i) {
            unique_labels.insert(labels[i]);
            if (labels[i] < min_label) min_label = labels[i];
            if (labels[i] > max_label) max_label = labels[i];
        }

        u32 K = static_cast<u32>(unique_labels.size());

        if (K < 2) {
            throw std::runtime_error("OvR requires at least 2 classes, found " + std::to_string(K));
        }

        if (K == 2) {
            // For K=2, we could optimize to single binary classifier, but for simplicity
            // and to match OvR semantics, we still train 2 heads
            // This also helps with K=2 parity testing
        }

        // Verify labels are contiguous from 0 to K-1 or min to max
        if (max_label - min_label + 1 != static_cast<i32>(K)) {
            throw std::runtime_error(
                "Labels must be contiguous. Found " + std::to_string(K) +
                " unique labels but range is [" + std::to_string(min_label) +
                ", " + std::to_string(max_label) + "]"
            );
        }

        // Initialize OvR model
        OvRModel model;
        model.num_classes = K;
        model.num_features = static_cast<u16>(dataset.cols());
        model.reserve_heads(K);

        // Train K binary classifiers (one for each class)
        if (parallel && K > 1) {
            // Parallel training using std::async
            std::vector<std::future<BoostModel>> futures;
            futures.reserve(K);

            for (u32 class_id = 0; class_id < K; ++class_id) {
                futures.push_back(std::async(std::launch::async, [&, class_id]() {
                    return train_binary_head(dataset, labels, min_label + static_cast<i32>(class_id),
                                            params, num_samples);
                }));
            }

            // Collect heads
            for (auto& future : futures) {
                model.add_head(future.get());
            }
        } else {
            // Sequential training
            for (u32 class_id = 0; class_id < K; ++class_id) {
                BoostModel head = train_binary_head(
                    dataset, labels, min_label + static_cast<i32>(class_id),
                    params, num_samples
                );
                model.add_head(std::move(head));
            }
        }

        return model;
    }

private:
    /// Train a single binary head for one class vs all others
    /// @param dataset      Training dataset
    /// @param labels       Original multiclass labels
    /// @param target_class Class to treat as positive (1), all others are negative (0)
    /// @param params       Training parameters
    /// @param num_samples  Number of samples
    /// @return Binary BoostModel
    static BoostModel train_binary_head(
        const Dataset& dataset,
        const i32* labels,
        i32 target_class,
        const BoostParams& params,
        usize num_samples
    ) {
        // Create binary labels: 1 if label == target_class, 0 otherwise
        std::vector<i32> binary_labels(num_samples);
        for (usize i = 0; i < num_samples; ++i) {
            binary_labels[i] = (labels[i] == target_class) ? 1 : 0;
        }

        // Create a modified dataset with binary labels
        // We'll temporarily replace labels in a copy
        Dataset binary_dataset = dataset.clone_with_labels(binary_labels.data());

        // Create params for this head with unique seed
        // Use XOR to generate deterministic but unique seeds per head
        BoostParams head_params = params;
        head_params.random_seed = params.random_seed ^ static_cast<u32>(target_class);
        head_params.task_type = 1;  // Binary classification

        // Train binary classifier
        return BoostTrainer::train(binary_dataset, head_params);
    }
};

} // namespace intgr
