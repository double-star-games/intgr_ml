#pragma once

#include "../intgrcore/Types.h"
#include <vector>
#include <string>
#include <fstream>
#include <cstring>

namespace intgr {

/// PCA model for dimensionality reduction
/// Stores principal components and mean vector in fixed-point
struct PCAModel {
    std::vector<i16> components;  ///< Principal components (n_components × n_features)
    std::vector<i16> mean;        ///< Feature means for centering
    i32 scale_factor;             ///< Scaling factor (default 256 = Q8.8)
    u16 num_features;             ///< Number of original features
    u16 num_components;           ///< Number of principal components
    f32 explained_variance_ratio; ///< Fraction of variance explained

    PCAModel()
        : scale_factor(256)
        , num_features(0)
        , num_components(0)
        , explained_variance_ratio(0.0f)
    {}

    /// Initialize with dimensions
    PCAModel(u16 n_features, u16 n_components)
        : components(static_cast<usize>(n_components) * n_features, 0)
        , mean(n_features, 0)
        , scale_factor(256)
        , num_features(n_features)
        , num_components(n_components)
        , explained_variance_ratio(0.0f)
    {}

    /// Get pointer to component i (row in component matrix)
    const i16* get_component(u16 component_idx) const noexcept {
        return components.data() + static_cast<usize>(component_idx) * num_features;
    }

    i16* get_component(u16 component_idx) noexcept {
        return components.data() + static_cast<usize>(component_idx) * num_features;
    }

    /// Estimate memory footprint in bytes
    usize memory_bytes() const noexcept {
        return sizeof(PCAModel)
             + components.size() * sizeof(i16)
             + mean.size() * sizeof(i16);
    }

    /// Save model to binary .srf file (IntgrReduce Format)
    bool save(const std::string& path) const;

    /// Load model from binary .srf file
    static PCAModel load(const std::string& path);
};

/// Binary file format header for .srf files (IntgrReduce Format)
struct SRFHeader {
    char magic[8];         ///< "SNAPMLv1"
    u16 model_type;        ///< 3=IntgrReduce
    u16 version;           ///< Format version (currently 1)
    u32 num_features;      ///< Number of original features
    u32 num_components;    ///< Number of principal components
    i32 scale_factor;      ///< Scaling factor
    f32 explained_var;     ///< Explained variance ratio
    u32 checksum;          ///< Simple checksum
    u8  reserved[4];       ///< Reserved for future use (36 bytes total)

    static constexpr u16 MODEL_TYPE_REDUCE = 3;
    static constexpr u16 FORMAT_VERSION = 1;

    SRFHeader()
        : model_type(MODEL_TYPE_REDUCE)
        , version(FORMAT_VERSION)
        , num_features(0)
        , num_components(0)
        , scale_factor(256)
        , explained_var(0.0f)
        , checksum(0)
    {
        std::memcpy(magic, "SNAPMLv1", 8);
        std::memset(reserved, 0, sizeof(reserved));
    }
};

static_assert(sizeof(SRFHeader) == 36, "SRFHeader must be 36 bytes for binary compatibility");

} // namespace intgr
