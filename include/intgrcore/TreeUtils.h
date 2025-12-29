#pragma once

#include "Types.h"
#include <vector>

namespace intgr {

/// Utilities for tree-based models (shared between IntgrBoost and IntgrForest)
/// These are template functions that work with any tree container
namespace TreeUtils {

/// Compute feature importance based on split counts across trees
/// @tparam TreeContainer Type of container holding trees (e.g., std::vector<Tree>)
/// @param trees Vector of decision trees
/// @param num_features Total number of features in the model
/// @return Vector of importance scores normalized to [0, 1] range
template<typename TreeContainer>
inline std::vector<f32> compute_feature_importance(
    const TreeContainer& trees,
    u16 num_features
) {
    std::vector<u32> split_counts(num_features, 0);

    // Count splits per feature across all trees
    for (const auto& tree : trees) {
        for (const auto& node : tree.nodes) {
            if (!node.is_leaf()) {
                if (node.feature < num_features) {
                    split_counts[node.feature]++;
                }
            }
        }
    }

    // Normalize to [0, 1] range
    u32 total_splits = 0;
    for (u32 count : split_counts) {
        total_splits += count;
    }

    std::vector<f32> importance(num_features, 0.0f);
    if (total_splits > 0) {
        for (usize i = 0; i < num_features; ++i) {
            importance[i] = static_cast<f32>(split_counts[i]) / static_cast<f32>(total_splits);
        }
    }

    return importance;
}

} // namespace TreeUtils
} // namespace intgr
