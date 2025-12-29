#pragma once

#include "Types.h"
#include <stdexcept>
#include <string>

namespace intgr {

/// System limits for intgr_ml v1
/// These are intentional design constraints for determinism, compact memory,
/// and edge deployability.
namespace limits {
    /// Maximum number of features supported by intgr_ml v1
    /// This limit enables:
    /// - Compact u16 feature indices in model serialization
    /// - Efficient memory layout on embedded/MCU targets
    /// - Deterministic behavior across all platforms
    ///
    /// For datasets exceeding this limit, consider:
    /// - Feature selection or dimensionality reduction (e.g., PCA)
    /// - Feature hashing to reduce dimensionality
    /// - Using a different ML framework designed for high-dimensional data
    constexpr u16 MAX_FEATURES = 65535;
}

/// Exception thrown when feature count exceeds MAX_FEATURES
class FeatureLimitError : public std::runtime_error {
public:
    explicit FeatureLimitError(usize actual_features)
        : std::runtime_error(make_message(actual_features))
        , actual_features_(actual_features)
    {}

    /// Get the actual feature count that exceeded the limit
    usize actual_features() const noexcept { return actual_features_; }

private:
    usize actual_features_;

    static std::string make_message(usize actual) {
        return "intgr_ml v1 supports up to " + std::to_string(limits::MAX_FEATURES) +
               " features; got " + std::to_string(actual);
    }
};

/// Check feature count and throw FeatureLimitError if exceeded
/// @param num_features  Number of features to validate
/// @throws FeatureLimitError if num_features > limits::MAX_FEATURES
inline void check_feature_limit(usize num_features) {
    if (num_features > limits::MAX_FEATURES) {
        throw FeatureLimitError(num_features);
    }
}

/// Check feature count and return success/failure
/// @param num_features  Number of features to validate
/// @return true if within limit, false if exceeded
inline bool is_feature_count_valid(usize num_features) noexcept {
    return num_features <= limits::MAX_FEATURES;
}

} // namespace intgr
