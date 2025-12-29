#pragma once

#include "../intgrcore/Types.h"
#include <vector>
#include <string>
#include <fstream>
#include <cstring>

namespace intgr {

/// Linear model (regression or logistic classification)
/// Stores weights and bias in fixed-point (int16)
struct LinearModel {
    std::vector<i16> weights;  ///< Weight vector (num_features × int16)
    i16 bias;                  ///< Bias term
    i32 scale_factor;          ///< Scaling factor for output (default 256 = Q8.8)
    u16 num_features;          ///< Number of input features
    u8  task_type;             ///< 0=regression, 1=logistic_classification

    // Task type constants
    static constexpr u8 TASK_REGRESSION = 0;
    static constexpr u8 TASK_LOGISTIC = 1;

    LinearModel()
        : bias(0)
        , scale_factor(256)
        , num_features(0)
        , task_type(TASK_REGRESSION)
    {}

    /// Initialize with given dimensions
    LinearModel(u16 n_features, u8 task)
        : weights(n_features, 0)
        , bias(0)
        , scale_factor(256)
        , num_features(n_features)
        , task_type(task)
    {}

    /// Get number of features
    usize size() const noexcept {
        return weights.size();
    }

    /// Estimate memory footprint in bytes
    usize memory_bytes() const noexcept {
        return sizeof(LinearModel) + weights.size() * sizeof(i16);
    }

    /// Save model to binary .slf file
    /// Format: magic header (8 bytes) + metadata + weights
    bool save(const std::string& path) const;

    /// Load model from binary .slf file
    static LinearModel load(const std::string& path);
};

/// Binary file format header for .slf files (IntgrLinear Format)
struct SLFHeader {
    char magic[8];       ///< "SNAPMLv1"
    u16 model_type;      ///< 2=IntgrLinear
    u16 version;         ///< Format version (currently 1)
    u32 num_features;    ///< Number of input features
    i32 scale_factor;    ///< Output scaling factor
    i16 bias;            ///< Bias term
    u8  task_type;       ///< Task type (regression/classification)
    u8  reserved[5];     ///< Reserved for future use
    u32 checksum;        ///< Simple checksum of weights
    i32 reserved_i32;    ///< Reserved for future use (alignment)

    static constexpr u16 MODEL_TYPE_LINEAR = 2;
    static constexpr u16 FORMAT_VERSION = 1;

    SLFHeader()
        : model_type(MODEL_TYPE_LINEAR)
        , version(FORMAT_VERSION)
        , num_features(0)
        , scale_factor(256)
        , bias(0)
        , task_type(0)
        , checksum(0)
        , reserved_i32(0)
    {
        std::memcpy(magic, "SNAPMLv1", 8);
        std::memset(reserved, 0, sizeof(reserved));
    }
};

static_assert(sizeof(SLFHeader) == 36, "SLFHeader must be 36 bytes for binary compatibility");

} // namespace intgr
