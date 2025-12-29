#pragma once

#include "EdgeModel.h"

/// IntgrEdge: Unified runtime for all IntgrML models
///
/// Simple unified interface:
///   EdgeModel model = EdgeModel::load("model.sbf");
///   i32 prediction = model.predict(features);
///
/// Supports:
///   - IntgrBoost (.sbf)
///   - IntgrForest (.sff)
///   - IntgrLinear (.slf)
///   - IntgrReduce (.srf)
///   - IntgrBayes (.sbn)
///
/// Auto-detects model type from file header.

namespace intgr {

/// Convenience function: Load model and predict single sample
inline i32 edge_predict(const std::string& model_path, const i8* features) {
    EdgeModel model = EdgeModel::load(model_path);
    return model.predict(features);
}

/// Convenience function: Load model and predict batch
inline void edge_predict_batch(const std::string& model_path,
                              const i8* features,
                              u32 num_samples,
                              i32* output) {
    EdgeModel model = EdgeModel::load(model_path);
    model.predict_batch(features, num_samples, output);
}

} // namespace intgr
