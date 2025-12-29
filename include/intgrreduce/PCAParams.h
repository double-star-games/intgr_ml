#pragma once

#include "../intgrcore/Types.h"

namespace intgr {

/// Parameters for PCA training
struct PCAParams {
    u16 num_components;      ///< Number of principal components (default: min(n_features, n_samples))
    u32 max_iterations;      ///< Max iterations for power iteration (default: 100)
    f32 tolerance;           ///< Convergence tolerance (default: 1e-4)
    u32 seed;                ///< Random seed for initialization (default: 1337)
    bool center_data;        ///< Center data by subtracting mean (default: true)
    bool verbose;            ///< Print training progress (default: false)

    PCAParams()
        : num_components(0)  // 0 = auto-detect
        , max_iterations(100)
        , tolerance(1e-4f)
        , seed(1337)
        , center_data(true)
        , verbose(false)
    {}

    // Fluent API for parameter setting
    PCAParams& components(u16 n) { num_components = n; return *this; }
    PCAParams& iterations(u32 iters) { max_iterations = iters; return *this; }
    PCAParams& tol(f32 t) { tolerance = t; return *this; }
    PCAParams& random_seed(u32 s) { seed = s; return *this; }
    PCAParams& center(bool c) { center_data = c; return *this; }
    PCAParams& print_progress(bool v) { verbose = v; return *this; }
};

} // namespace intgr
