#pragma once

#include "../intgrcore/Types.h"
#include <vector>
#include <string>
#include <fstream>
#include <cstring>

namespace intgr {

/// Naive Bayes model for classification
/// Stores log probabilities in fixed-point (int16)
/// Uses log-domain arithmetic to avoid underflow
struct BayesModel {
    std::vector<i16> class_log_priors;  ///< Log P(class) for each class
    std::vector<i16> feature_log_probs; ///< Log P(feature|class) [class][feature][value]
    i32 scale_factor;                   ///< Scaling factor (default 256 = Q8.8)
    u16 num_classes;                    ///< Number of classes
    u16 num_features;                   ///< Number of features
    u16 num_bins;                       ///< Number of bins per feature (default 256)

    BayesModel()
        : scale_factor(256)
        , num_classes(0)
        , num_features(0)
        , num_bins(256)
    {}

    /// Initialize with dimensions
    BayesModel(u16 n_classes, u16 n_features, u16 n_bins = 256)
        : class_log_priors(n_classes, 0)
        , feature_log_probs(static_cast<usize>(n_classes) * n_features * n_bins, 0)
        , scale_factor(256)
        , num_classes(n_classes)
        , num_features(n_features)
        , num_bins(n_bins)
    {}

    /// Get log probability: log P(feature_val | class)
    /// @param class_idx   Class index
    /// @param feature_idx Feature index
    /// @param value       Feature value (quantized to [0, 255])
    i16 get_log_prob(u16 class_idx, u16 feature_idx, u8 value) const noexcept {
        usize idx = static_cast<usize>(class_idx) * num_features * num_bins
                  + static_cast<usize>(feature_idx) * num_bins
                  + value;
        return feature_log_probs[idx];
    }

    /// Set log probability
    void set_log_prob(u16 class_idx, u16 feature_idx, u8 value, i16 log_prob) noexcept {
        usize idx = static_cast<usize>(class_idx) * num_features * num_bins
                  + static_cast<usize>(feature_idx) * num_bins
                  + value;
        feature_log_probs[idx] = log_prob;
    }

    /// Estimate memory footprint in bytes
    usize memory_bytes() const noexcept {
        return sizeof(BayesModel)
             + class_log_priors.size() * sizeof(i16)
             + feature_log_probs.size() * sizeof(i16);
    }

    /// Save model to binary .sbn file (IntgrBayes Format)
    bool save(const std::string& path) const;

    /// Load model from binary .sbn file
    static BayesModel load(const std::string& path);
};

/// Binary file format header for .sbn files (IntgrBayes Format)
struct SBNHeader {
    char magic[8];        ///< "SNAPMLv1"
    u16 model_type;       ///< 4=IntgrBayes
    u16 version;          ///< Format version (currently 1)
    u16 num_classes;      ///< Number of classes
    u16 num_features;     ///< Number of features
    u16 num_bins;         ///< Number of bins per feature
    u8  reserved[2];      ///< Reserved for alignment
    i32 scale_factor;     ///< Scaling factor
    u32 checksum;         ///< Simple checksum
    u8  reserved2[8];     ///< Reserved (36 bytes total)

    static constexpr u16 MODEL_TYPE_BAYES = 4;
    static constexpr u16 FORMAT_VERSION = 1;

    SBNHeader()
        : model_type(MODEL_TYPE_BAYES)
        , version(FORMAT_VERSION)
        , num_classes(0)
        , num_features(0)
        , num_bins(256)
        , scale_factor(256)
        , checksum(0)
    {
        std::memcpy(magic, "SNAPMLv1", 8);
        std::memset(reserved, 0, sizeof(reserved));
        std::memset(reserved2, 0, sizeof(reserved2));
    }
};

static_assert(sizeof(SBNHeader) == 36, "SBNHeader must be 36 bytes for binary compatibility");

} // namespace intgr
