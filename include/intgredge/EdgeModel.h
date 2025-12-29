#pragma once

#include "../intgrcore/Types.h"
#include "../intgrboost/BoostModel.h"
#include "../intgrforest/ForestModel.h"
#include "../intgrlinear/LinearModel.h"
#include "../intgrreduce/PCAModel.h"
#include "../intgrbayes/BayesModel.h"
#include <string>
#include <variant>
#include <fstream>
#include <cstring>

namespace intgr {

/// Unified model container for all IntgrML model types
/// Provides single interface for loading and inference
class EdgeModel {
public:
    /// Model type enumeration
    enum class Type {
        Boost = 0,
        Forest = 1,
        Linear = 2,
        Reduce = 3,
        Bayes = 4,
        Unknown = 255
    };

    /// Default constructor (empty model)
    EdgeModel() : type_(Type::Unknown) {}

    /// Construct from specific model type
    explicit EdgeModel(const BoostModel& model)
        : type_(Type::Boost), model_(model) {}

    explicit EdgeModel(const ForestModel& model)
        : type_(Type::Forest), model_(model) {}

    explicit EdgeModel(const LinearModel& model)
        : type_(Type::Linear), model_(model) {}

    explicit EdgeModel(const PCAModel& model)
        : type_(Type::Reduce), model_(model) {}

    explicit EdgeModel(const BayesModel& model)
        : type_(Type::Bayes), model_(model) {}

    /// Load model from file (auto-detects type)
    static EdgeModel load(const std::string& path);

    /// Get model type
    Type type() const noexcept { return type_; }

    /// Check if model is loaded
    bool is_loaded() const noexcept { return type_ != Type::Unknown; }

    /// Get model type as string
    const char* type_name() const noexcept;

    /// Predict single sample (regression or classification)
    /// Returns int32 prediction value
    i32 predict(const i8* features) const;

    /// Predict probability for classification models
    /// Returns probability in [0, 255] range
    u8 predict_proba(const i8* features) const;

    /// Batch prediction (writes to output array)
    void predict_batch(const i8* features, u32 num_samples, i32* output) const;

    /// Get underlying model (throws if wrong type)
    const BoostModel& as_boost() const;
    const ForestModel& as_forest() const;
    const LinearModel& as_linear() const;
    const PCAModel& as_reduce() const;
    const BayesModel& as_bayes() const;

    /// Get number of features
    u16 num_features() const noexcept;

    /// Estimate memory footprint
    usize memory_bytes() const noexcept;

private:
    Type type_;
    std::variant<
        std::monostate,
        BoostModel,
        ForestModel,
        LinearModel,
        PCAModel,
        BayesModel
    > model_;

    /// Detect model type from file header
    static Type detect_type(const std::string& path);
};

} // namespace intgr
