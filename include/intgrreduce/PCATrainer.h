#pragma once

#include "PCAModel.h"
#include "PCAParams.h"
#include "../intgrdata/Dataset.h"
#include <vector>

namespace intgr {

/// PCA trainer using power iteration for eigenvector computation
class PCATrainer {
public:
    /// Train PCA model from dataset
    /// @param dataset  Training data
    /// @param params   Training parameters
    /// @return Trained PCA model
    static PCAModel train(const Dataset& dataset, const PCAParams& params);

private:
    /// Compute feature means
    static std::vector<f32> compute_means(const Dataset& dataset);

    /// Compute covariance matrix in float (for numerical stability)
    /// Returns flattened covariance matrix (n_features × n_features)
    static std::vector<f32> compute_covariance(const Dataset& dataset,
                                               const std::vector<f32>& means);

    /// Find largest eigenvector using power iteration
    /// @param cov_matrix  Covariance matrix (n × n, row-major)
    /// @param n           Matrix dimension
    /// @param max_iters   Maximum iterations
    /// @param tolerance   Convergence tolerance
    /// @param seed        Random seed for initialization
    /// @param eigenvalue  Output: largest eigenvalue
    /// @return Eigenvector corresponding to largest eigenvalue
    static std::vector<f32> power_iteration(const std::vector<f32>& cov_matrix,
                                            u32 n,
                                            u32 max_iters,
                                            f32 tolerance,
                                            u32& seed,
                                            f32& eigenvalue);

    /// Deflate covariance matrix by removing component
    /// C' = C - λ * v * v^T
    static void deflate_matrix(std::vector<f32>& cov_matrix,
                              const std::vector<f32>& eigenvector,
                              f32 eigenvalue,
                              u32 n);

    /// Normalize vector to unit length
    static void normalize(std::vector<f32>& vec);

    /// Quantize float vector to int16 with given scale
    static std::vector<i16> quantize_vector(const std::vector<f32>& vec,
                                           i32 scale_factor);

    /// Compute total variance for explained variance ratio
    static f32 compute_total_variance(const std::vector<f32>& cov_matrix, u32 n);
};

} // namespace intgr
