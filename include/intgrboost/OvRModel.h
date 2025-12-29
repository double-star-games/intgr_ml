#pragma once

#include "BoostModel.h"
#include "../intgrcore/Types.h"
#include <vector>
#include <string>
#include <fstream>

namespace intgr {

/// One-vs-Rest multiclass model using K binary IntgrBoost classifiers
/// Each "head" is a binary classifier trained to distinguish one class from all others
struct OvRModel {
    std::vector<BoostModel> heads;  ///< K binary boosters (one per class)
    u32 num_classes;                ///< Number of classes (K)
    u16 num_features;               ///< Number of input features

    /// Optional Platt scaling parameters for probability calibration
    /// Each pair is (a_q16, b_q16) where prob = sigmoid(a*score + b)
    /// Stored in Q16.16 fixed-point format
    std::vector<std::pair<i32, i32>> platt_params;
    bool has_platt_scaling;         ///< Whether Platt scaling is enabled

    OvRModel()
        : num_classes(0)
        , num_features(0)
        , has_platt_scaling(false)
    {}

    /// Get number of heads (should equal num_classes)
    usize num_heads() const noexcept {
        return heads.size();
    }

    /// Reserve space for K heads
    void reserve_heads(u32 K) {
        heads.reserve(K);
        platt_params.reserve(K);
    }

    /// Add a binary head for a specific class
    void add_head(BoostModel&& head) {
        heads.push_back(std::move(head));
    }

    /// Save model to binary .ovr file
    /// Format: OVR header + K BoostModel chunks
    bool save(const std::string& path) const;

    /// Load model from binary .ovr file
    static OvRModel load(const std::string& path);

    /// Get total number of nodes across all heads
    usize total_nodes() const noexcept {
        usize count = 0;
        for (const auto& head : heads) {
            count += head.total_nodes();
        }
        return count;
    }

    /// Estimate memory footprint in bytes
    usize memory_bytes() const noexcept {
        usize total = sizeof(OvRModel);
        for (const auto& head : heads) {
            total += head.memory_bytes();
        }
        return total;
    }

    /// Compute feature importance averaged across all heads
    /// Returns vector of size num_features with importance scores (0.0 to 1.0)
    std::vector<f32> compute_feature_importance() const {
        if (heads.empty()) {
            return std::vector<f32>(num_features, 0.0f);
        }

        // Average importance across all heads
        std::vector<f32> importance(num_features, 0.0f);
        for (const auto& head : heads) {
            auto head_importance = head.compute_feature_importance();
            for (usize i = 0; i < num_features && i < head_importance.size(); ++i) {
                importance[i] += head_importance[i];
            }
        }

        // Normalize by number of heads
        f32 scale = 1.0f / static_cast<f32>(heads.size());
        for (auto& imp : importance) {
            imp *= scale;
        }

        return importance;
    }
};

/// Extended metadata for OvR model provenance
struct OvRMetadata {
    char version[16];          ///< IntgrML version (e.g., "1.2.0")
    char commit_hash[8];       ///< Git commit hash (7 chars + null)
    char accumulator[16];      ///< Accumulator format (e.g., "Q16.16")
    u8   deterministic;        ///< 1 if training was deterministic
    u8   has_platt_scaling;    ///< 1 if Platt scaling parameters present
    u8   reserved[14];         ///< Reserved for future use

    OvRMetadata()
        : deterministic(1)
        , has_platt_scaling(0)
    {
        std::memcpy(version, "1.2.0", 6);
        std::memcpy(commit_hash, "dev", 4);
        std::memcpy(accumulator, "Q16.16", 7);
        std::memset(reserved, 0, sizeof(reserved));
    }
};

static_assert(sizeof(OvRMetadata) == 56, "OvRMetadata must be 56 bytes");

/// Binary file format header for .ovr files
struct OvRHeader {
    char magic[8];       ///< "SNAPOVRv"
    u16 model_type;      ///< Always 1 for OvR
    u16 version;         ///< Format version (v1)
    u32 num_classes;     ///< Number of classes (K)
    u32 num_features;    ///< Number of input features
    u32 caps;            ///< Capability flags (bit 0=has_platt_scaling)
    u32 checksum;        ///< Simple checksum of data
    u8  reserved[12];    ///< Reserved for future use (12 bytes to reach 40 total)

    static constexpr u16 MODEL_TYPE_OVR = 1;
    static constexpr u16 FORMAT_VERSION = 1;

    // Capability flags
    static constexpr u32 CAP_PLATT_SCALING = 0x01;  // Has Platt scaling parameters

    OvRHeader()
        : model_type(MODEL_TYPE_OVR)
        , version(FORMAT_VERSION)
        , num_classes(0)
        , num_features(0)
        , caps(0)
        , checksum(0)
    {
        std::memcpy(magic, "SNAPOVRv", 8);
        std::memset(reserved, 0, sizeof(reserved));
    }
};

static_assert(sizeof(OvRHeader) == 40, "OvRHeader must be 40 bytes");

} // namespace intgr
