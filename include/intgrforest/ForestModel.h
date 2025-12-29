#pragma once

#include "../intgrcore/Types.h"
#include "../intgrcore/TreeUtils.h"
#include "../intgrboost/BoostModel.h"
#include <vector>
#include <string>

namespace intgr {

/// Random Forest model
/// Ensemble of decision trees with bagging and feature randomization
/// Uses IntgrBoost's Tree structure internally
struct ForestModel {
    std::vector<Tree> trees;    ///< Ensemble of decision trees
    i32 scale_factor;           ///< Scaling factor for averaging (e.g., 256/num_trees)
    u16 num_features;           ///< Number of input features
    u8  task_type;              ///< 0=regression, 1=binary_classification

    // Task type constants
    static constexpr u8 TASK_REGRESSION = 0;
    static constexpr u8 TASK_BINARY_CLASSIFICATION = 1;

    ForestModel()
        : scale_factor(256)
        , num_features(0)
        , task_type(TASK_REGRESSION)
    {}

    /// Get number of trees in forest
    usize num_trees() const noexcept {
        return trees.size();
    }

    /// Reserve space for trees
    void reserve_trees(usize count) {
        trees.reserve(count);
    }

    /// Add tree to ensemble
    void add_tree(Tree&& tree) {
        trees.push_back(std::move(tree));
    }

    /// Save model to binary .sff file
    /// Format: magic header (8 bytes) + metadata + trees
    bool save(const std::string& path) const;

    /// Load model from binary .sff file
    static ForestModel load(const std::string& path);

    /// Get total number of nodes across all trees
    usize total_nodes() const noexcept {
        usize count = 0;
        for (const auto& tree : trees) {
            count += tree.size();
        }
        return count;
    }

    /// Estimate memory footprint in bytes
    usize memory_bytes() const noexcept {
        return sizeof(ForestModel) + total_nodes() * sizeof(Node);
    }

    /// Compute feature importance based on split counts
    std::vector<f32> compute_feature_importance() const {
        return TreeUtils::compute_feature_importance(trees, num_features);
    }
};

/// Binary file format header for .sff files
struct SFFHeader {
    char magic[8];       ///< "SNAPMLv1"
    u16 model_type;      ///< 1=IntgrForest
    u16 version;         ///< Format version (currently 1)
    u32 num_trees;       ///< Number of trees in forest
    u32 num_features;    ///< Number of input features
    i32 scale_factor;    ///< Averaging scale factor
    i32 reserved_int;    ///< Reserved for future use (maintains 36-byte alignment)
    u8  task_type;       ///< Task type (regression/classification)
    u8  reserved[3];     ///< Reserved for future use
    u32 checksum;        ///< Simple checksum of data

    static constexpr u16 MODEL_TYPE_FOREST = 1;
    static constexpr u16 FORMAT_VERSION = 1;

    SFFHeader()
        : model_type(MODEL_TYPE_FOREST)
        , version(FORMAT_VERSION)
        , num_trees(0)
        , num_features(0)
        , scale_factor(256)
        , reserved_int(0)
        , task_type(0)
        , checksum(0)
    {
        std::memcpy(magic, "SNAPMLv1", 8);
        std::memset(reserved, 0, sizeof(reserved));
    }
};

static_assert(sizeof(SFFHeader) == 36, "SFFHeader must be 36 bytes for binary compatibility");

} // namespace intgr
