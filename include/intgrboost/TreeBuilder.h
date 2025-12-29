#pragma once

#include "BoostModel.h"
#include "BoostParams.h"
#include "Histogram.h"
#include "SplitFinder.h"
#include "../intgrdata/Dataset.h"
#include <vector>
#include <algorithm>

namespace intgr {

/// Builds a single decision tree for gradient boosting
class TreeBuilder {
public:
    explicit TreeBuilder(const BoostParams& params)
        : params_(params)
    {}

    /// Build a decision tree on a subset of samples
    /// @param dataset      Quantized feature dataset
    /// @param gradients    Gradient values (int32 array, length = dataset.rows())
    /// @param hessians     Hessian values (int32 array, length = dataset.rows())
    /// @param sample_mask  Boolean mask (1=include, 0=exclude), length = dataset.rows()
    /// @return Constructed tree
    Tree build(
        const Dataset& dataset,
        const i32* gradients,
        const i32* hessians,
        const u8* sample_mask
    ) {
        Tree tree;

        // Build sample index list from mask
        std::vector<u32> sample_indices;
        for (usize i = 0; i < dataset.rows(); ++i) {
            if (sample_mask[i]) {
                sample_indices.push_back(static_cast<u32>(i));
            }
        }

        // Build tree recursively
        build_recursive(tree, dataset, gradients, hessians, sample_indices, 0);

        return tree;
    }

private:
    const BoostParams& params_;

    /// Recursive tree building
    /// @return Node index of the created node
    u32 build_recursive(
        Tree& tree,
        const Dataset& dataset,
        const i32* gradients,
        const i32* hessians,
        const std::vector<u32>& sample_indices,
        u32 depth
    ) {
        usize num_samples = sample_indices.size();

        // Stopping criteria: max depth or insufficient samples
        if (depth >= params_.max_depth ||
            num_samples < 2 * static_cast<usize>(params_.min_samples_leaf)) {
            return create_leaf_node(tree, gradients, hessians, sample_indices);
        }

        // Find best split across all features
        SplitResult best_split;
        u32 best_feature = 0;
        find_best_split_across_features(dataset, gradients, hessians, sample_indices,
                                        best_split, best_feature);

        // If no valid split found, create leaf
        if (!best_split.found) {
            return create_leaf_node(tree, gradients, hessians, sample_indices);
        }

        // Partition samples based on split
        std::vector<u32> left_indices, right_indices;
        partition_samples(dataset, sample_indices, best_feature, best_split.threshold,
                          best_split.left_count, best_split.right_count,
                          left_indices, right_indices);

        // Create split node
        u32 current_node_idx = create_split_node(tree, best_feature, best_split.threshold);

        // Build left and right subtrees
        u32 left_child_idx = build_recursive(tree, dataset, gradients, hessians, left_indices, depth + 1);
        u32 right_child_idx = build_recursive(tree, dataset, gradients, hessians, right_indices, depth + 1);

        // Update current node with correct child indices
        tree.nodes[current_node_idx].left = static_cast<i32>(left_child_idx);
        tree.nodes[current_node_idx].right = static_cast<i32>(right_child_idx);

        return current_node_idx;
    }

    /// Find best split across all features
    void find_best_split_across_features(
        const Dataset& dataset,
        const i32* gradients,
        const i32* hessians,
        const std::vector<u32>& sample_indices,
        SplitResult& best_split,
        u32& best_feature
    ) const {
        for (usize feat = 0; feat < dataset.cols(); ++feat) {
            // Build histogram for this feature
            Histogram hist;
            hist.clear();

            for (u32 idx : sample_indices) {
                i8 feature_val = dataset.get_i8(idx, feat);
                hist.add(feature_val, gradients[idx], hessians[idx]);
            }

            // Find best split for this feature with regularization
            SplitResult split = find_best_split(
                hist,
                params_.min_samples_leaf,
                params_.lambda,
                params_.min_gain,
                params_.gamma
            );

            if (split.found && split.gain > best_split.gain) {
                best_split = split;
                best_feature = static_cast<u32>(feat);
            }
        }
    }

    /// Partition samples into left and right based on split threshold
    void partition_samples(
        const Dataset& dataset,
        const std::vector<u32>& sample_indices,
        u32 feature,
        i16 threshold,
        i32 expected_left_count,
        i32 expected_right_count,
        std::vector<u32>& left_indices,
        std::vector<u32>& right_indices
    ) const {
        // expected counts are always non-negative (they're sample counts)
        left_indices.reserve(expected_left_count > 0 ? static_cast<usize>(expected_left_count) : 0);
        right_indices.reserve(expected_right_count > 0 ? static_cast<usize>(expected_right_count) : 0);

        for (u32 idx : sample_indices) {
            i8 feature_val = dataset.get_i8(idx, feature);

            if (feature_val <= threshold) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        }
    }

    /// Create split node with appropriate threshold precision
    u32 create_split_node(Tree& tree, u32 feature, i16 threshold) {
        u32 node_idx = static_cast<u32>(tree.size());

        // Choose node type based on precision policy and threshold value
        bool use_i16_threshold = false;
        if (params_.precision.threshold == PrecisionPolicy::Threshold::I16) {
            use_i16_threshold = true;
        } else if (params_.precision.threshold == PrecisionPolicy::Threshold::Auto) {
            use_i16_threshold = (threshold < -128 || threshold > 127);
        }

        if (use_i16_threshold) {
            tree.add_node(Node::make_split_i16(
                static_cast<u16>(feature), threshold, 0, 0
            ));
        } else {
            tree.add_node(Node::make_split(
                static_cast<u16>(feature), static_cast<i8>(threshold), 0, 0
            ));
        }

        return node_idx;
    }

    /// Create leaf node with calculated value
    u32 create_leaf_node(
        Tree& tree,
        const i32* gradients,
        const i32* hessians,
        const std::vector<u32>& sample_indices
    ) const {
        i32 leaf_value = calculate_leaf_value(gradients, hessians, sample_indices);
        u32 node_idx = static_cast<u32>(tree.size());
        tree.add_node(Node::make_leaf(leaf_value));
        return node_idx;
    }

    /// Calculate leaf value using gradient boosting formula
    /// leaf_value = -sum(gradients) / (sum(hessians) + lambda)
    i32 calculate_leaf_value(
        const i32* gradients,
        const i32* hessians,
        const std::vector<u32>& sample_indices
    ) const {
        i64 grad_sum = 0;
        i64 hess_sum = 0;

        for (u32 idx : sample_indices) {
            grad_sum += gradients[idx];
            hess_sum += hessians[idx];
        }

        // Add L2 regularization to denominator
        i64 denom = hess_sum + params_.lambda;

        if (denom == 0) {
            return 0;  // Avoid division by zero
        }

        // leaf_value = -grad_sum / denom
        // Note: Previous implementation scaled numerator by 256 for fixed-point arithmetic.
        // This scaling is no longer needed because the Q16.16 accumulator eliminates
        // the need for leaf value scaling during training.
        return static_cast<i32>(-grad_sum / denom);
    }
};

} // namespace intgr
