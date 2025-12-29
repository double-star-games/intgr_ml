#pragma once

#include "../intgrcore/Types.h"
#include "../intgrcore/TreeUtils.h"
#include <vector>
#include <string>
#include <fstream>
#include <cstring>

namespace intgr {

/// Decision tree node (16-byte aligned for cache efficiency)
/// Mixed-precision layout with tagged threshold storage
/// Supports both int8 and int16 thresholds via flag bit
struct alignas(16) Node {
    i32 left;          ///< Left child index OR leaf value (if is_leaf)  [bytes 0-3]
    i32 right;         ///< Right child index (unused if is_leaf)        [bytes 4-7]
    u16 feature;       ///< Feature column index                         [bytes 8-9]
    u8  flags;         ///< Flags: bit0=is_leaf, bit1=missing_left, bit2=thresh_is_i16  [byte 10]
    i8  thresh_i8;     ///< Int8 threshold (valid if !(flags & FLAG_THRESH_I16))  [byte 11]
    i16 thresh_i16;    ///< Int16 threshold (valid if (flags & FLAG_THRESH_I16))  [bytes 12-13]
    i16 leaf_idx;      ///< Index into leaf table; -1 for splits (reserved for Phase 2)  [bytes 14-15]

    // Flag bits
    static constexpr u8 FLAG_IS_LEAF = 0x01;
    static constexpr u8 FLAG_MISSING_LEFT = 0x02;
    static constexpr u8 FLAG_THRESH_I16 = 0x04;  // NEW: threshold is int16 (not int8)

    /// Check if this node is a leaf
    bool is_leaf() const noexcept {
        return (flags & FLAG_IS_LEAF) != 0;
    }

    /// Get leaf value (only valid if is_leaf())
    i32 leaf_value() const noexcept {
        return left;  // Reuse left field for leaf value
    }

    /// Get threshold value (handles both int8 and int16 thresholds)
    /// Returns int16 to accommodate both types
    i16 get_threshold() const noexcept {
        if (flags & FLAG_THRESH_I16) {
            return thresh_i16;
        } else {
            return static_cast<i16>(thresh_i8);
        }
    }

    /// Create leaf node
    static Node make_leaf(i32 value) noexcept {
        Node n;
        n.left = value;   // Store leaf value in left
        n.right = 0;
        n.feature = 0;
        n.flags = FLAG_IS_LEAF;
        n.thresh_i8 = 0;
        n.thresh_i16 = 0;
        n.leaf_idx = -1;
        return n;
    }

    /// Create split node with int8 threshold (default, current behavior)
    static Node make_split(u16 feature_idx, i8 threshold_val, i32 left_idx, i32 right_idx) noexcept {
        Node n;
        n.left = left_idx;
        n.right = right_idx;
        n.feature = feature_idx;
        n.flags = 0;  // Not a leaf, int8 threshold
        n.thresh_i8 = threshold_val;
        n.thresh_i16 = 0;  // Unused
        n.leaf_idx = -1;
        return n;
    }

    /// Create split node with int16 threshold (NEW: mixed-precision)
    static Node make_split_i16(u16 feature_idx, i16 threshold_val, i32 left_idx, i32 right_idx) noexcept {
        Node n;
        n.left = left_idx;
        n.right = right_idx;
        n.feature = feature_idx;
        n.flags = FLAG_THRESH_I16;  // Mark as int16 threshold
        n.thresh_i8 = 0;  // Unused
        n.thresh_i16 = threshold_val;
        n.leaf_idx = -1;
        return n;
    }
};

static_assert(sizeof(Node) == 16, "Node must be exactly 16 bytes");

/// Single decision tree
struct Tree {
    std::vector<Node> nodes;  ///< Nodes in breadth-first order (root at index 0)

    /// Get root node
    const Node& root() const noexcept {
        return nodes[0];
    }

    /// Get number of nodes
    usize size() const noexcept {
        return nodes.size();
    }

    /// Check if tree is empty
    bool empty() const noexcept {
        return nodes.empty();
    }

    /// Reserve capacity
    void reserve(usize capacity) {
        nodes.reserve(capacity);
    }

    /// Add node and return its index
    u32 add_node(const Node& node) {
        nodes.push_back(node);
        return static_cast<u32>(nodes.size() - 1);
    }
};

/// Complete gradient boosting model
struct BoostModel {
    std::vector<Tree> trees;    ///< Ensemble of decision trees
    i32 bias_fp;                ///< Global bias (fixed-point)
    i32 learning_rate_fp;       ///< Learning rate (fixed-point, e.g., FIXED_POINT_SCALE = 1.0)
    u16 num_features;           ///< Number of input features
    u8  task_type;              ///< 0=regression, 1=binary_classification

    // Task type constants
    static constexpr u8 TASK_REGRESSION = 0;
    static constexpr u8 TASK_BINARY_CLASSIFICATION = 1;

    BoostModel()
        : bias_fp(0)
        , learning_rate_fp(constants::FIXED_POINT_SCALE)  // Default: 1.0 in fixed-point
        , num_features(0)
        , task_type(TASK_REGRESSION)
    {}

    /// Get number of trees
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

    /// Save model to binary .sbf file
    /// Format: magic header (8 bytes) + metadata + trees
    bool save(const std::string& path) const;

    /// Load model from binary .sbf file
    static BoostModel load(const std::string& path);

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
        return sizeof(BoostModel) + total_nodes() * sizeof(Node);
    }

    /// Compute feature importance based on split counts
    std::vector<f32> compute_feature_importance() const {
        return TreeUtils::compute_feature_importance(trees, num_features);
    }
};

/// Extended metadata for model provenance and configuration
/// Only written if CAP_HAS_METADATA flag is set
struct ModelMetadata {
    char version[16];          ///< IntgrML version (e.g., "1.1.0")
    char commit_hash[8];       ///< Git commit hash (7 chars + null)
    char accumulator[16];      ///< Accumulator format (e.g., "Q16.16")
    u8   deterministic;        ///< 1 if training was deterministic
    u8   reserved[15];         ///< Reserved for future use

    ModelMetadata()
        : deterministic(1)
    {
        // Zero-initialize all arrays to ensure deterministic serialization
        std::memset(version, 0, sizeof(version));
        std::memset(commit_hash, 0, sizeof(commit_hash));
        std::memset(accumulator, 0, sizeof(accumulator));
        std::memset(reserved, 0, sizeof(reserved));

        // Copy strings (now into zero-filled arrays)
        std::memcpy(version, "0.1.0", 6);
        std::memcpy(commit_hash, "1c36df5", 8);
        std::memcpy(accumulator, "Q16.16", 7);
    }
};

static_assert(sizeof(ModelMetadata) == 56, "ModelMetadata must be 56 bytes");

/// Binary file format header for .sbf files
struct SBFHeader {
    char magic[8];       ///< "SNAPMLv1"
    u16 model_type;      ///< 0=IntgrBoost
    u16 version;         ///< Format version (v2 = mixed-precision support)
    u32 num_trees;       ///< Number of trees in model
    u32 num_features;    ///< Number of input features
    i32 bias_fp;         ///< Global bias
    i32 learning_rate_fp; ///< Learning rate
    u8  task_type;       ///< Task type (regression/classification)
    u8  reserved[3];     ///< Reserved for future use
    u32 caps;            ///< Capability flags (bit 0=int16_thresholds, bit 1=int32_leaves, etc)
    u32 checksum;        ///< Simple checksum of data

    static constexpr u16 MODEL_TYPE_BOOST = 0;
    static constexpr u16 FORMAT_VERSION = 2;  // v2 adds mixed-precision support

    // Capability flags
    static constexpr u32 CAP_INT16_THRESHOLDS = 0x01;  // Model uses int16 thresholds
    static constexpr u32 CAP_INT32_LEAVES     = 0x02;  // Model uses int32 leaf values
    static constexpr u32 CAP_PER_FEATURE_PREC = 0x04;  // Per-feature precision enabled
    static constexpr u32 CAP_PRED_ACCUM_Q16_16 = 0x08; // Training used Q16.16 accumulator (v1.1+)
    static constexpr u32 CAP_HAS_METADATA     = 0x10;  // Extended metadata section present

    SBFHeader()
        : model_type(MODEL_TYPE_BOOST)
        , version(FORMAT_VERSION)
        , num_trees(0)
        , num_features(0)
        , bias_fp(0)
        , learning_rate_fp(constants::FIXED_POINT_SCALE)
        , task_type(0)
        , caps(0)           // No capabilities by default (int8-only mode)
        , checksum(0)
    {
        std::memcpy(magic, "SNAPMLv1", 8);
        std::memset(reserved, 0, sizeof(reserved));
    }
};

static_assert(sizeof(SBFHeader) == 40, "SBFHeader must be 40 bytes (v2 format)");

} // namespace intgr
