#pragma once

#include "PCAModel.h"
#include "../intgrcore/Types.h"

namespace intgr {

/// PCA inference engine
/// Performs dimensionality reduction using principal components
class PCA {
public:
    /// Transform a single sample to reduced dimension
    /// @param model     Trained PCA model
    /// @param features  Input features (int8, length = num_features)
    /// @param output    Output reduced features (int16, length = num_components)
    static void transform(const PCAModel& model,
                         const i8* features,
                         i16* output) noexcept;

    /// Transform multiple samples (batch)
    /// @param model       Trained PCA model
    /// @param features    Input matrix (row-major, num_samples × num_features)
    /// @param num_samples Number of samples
    /// @param output      Output matrix (row-major, num_samples × num_components)
    static void transform_batch(const PCAModel& model,
                                const i8* features,
                                u32 num_samples,
                                i16* output) noexcept;

    /// Inverse transform from reduced dimension back to original
    /// @param model      Trained PCA model
    /// @param reduced    Reduced features (int16, length = num_components)
    /// @param output     Reconstructed features (int16, length = num_features)
    static void inverse_transform(const PCAModel& model,
                                  const i16* reduced,
                                  i16* output) noexcept;

    /// Compute reconstruction error for quality assessment
    /// @param model       Trained PCA model
    /// @param features    Original features
    /// @param num_samples Number of samples
    /// @return Mean squared reconstruction error
    static f32 reconstruction_error(const PCAModel& model,
                                   const i8* features,
                                   u32 num_samples) noexcept;

private:
    /// Dot product: (features - mean) · component
    /// Returns int32 scaled result
    static i32 centered_dot_product(const PCAModel& model,
                                   const i8* features,
                                   u16 component_idx) noexcept;
};

} // namespace intgr
