#pragma once

#include "../intgrcore/Types.h"

namespace intgr {

/// Parameters for Naive Bayes training
struct BayesParams {
    f32 alpha;          ///< Laplace smoothing parameter (default: 1.0)
    u16 num_bins;       ///< Number of bins for quantized features (default: 256)
    bool verbose;       ///< Print training progress (default: false)

    BayesParams()
        : alpha(1.0f)
        , num_bins(256)
        , verbose(false)
    {}

    // Fluent API for parameter setting
    BayesParams& smoothing(f32 a) { alpha = a; return *this; }
    BayesParams& bins(u16 b) { num_bins = b; return *this; }
    BayesParams& print_progress(bool v) { verbose = v; return *this; }
};

} // namespace intgr
