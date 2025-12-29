#pragma once

#include "../intgrcore/Types.h"

namespace intgr {

/// Precision policy for mixed-precision gradient boosting
/// Controls which components use higher-bit integers (int16/int32)
/// vs standard int8 representation
///
/// Default values maintain bit-exact compatibility with current int8-only engine
struct PrecisionPolicy {
    /// Threshold precision: controls split threshold storage
    enum class Threshold : u8 {
        I8,    ///< Standard int8 thresholds (default, current behavior)
        I16,   ///< Higher-precision int16 thresholds
        Auto   ///< Automatically select based on feature variance
    };

    /// Residual quantization precision (per boosting round)
    /// Controls gradient/hessian target quantization
    enum class Residual : u8 {
        Q8_8,    ///< 8-bit fractional (default, current behavior)
        Q12_4,   ///< 12-bit integer, 4-bit fractional
        Q16_16   ///< 16-bit integer, 16-bit fractional (max precision)
    };

    /// Leaf value storage precision
    enum class Leaf : u8 {
        I16,   ///< 16-bit leaf values
        I32,   ///< 32-bit leaf values (default, current behavior)
        Auto   ///< Automatically select based on tree depth/variance
    };

    /// Accumulator precision for prediction sum
    enum class Accum : u8 {
        I32,   ///< 32-bit accumulator (default, current behavior)
        I64    ///< 64-bit accumulator (for very deep ensembles)
    };

    // Precision settings (defaults = int8-only mode)
    Threshold threshold = Threshold::I8;
    Residual  residual  = Residual::Q8_8;
    Leaf      leaf      = Leaf::I32;
    Accum     accum     = Accum::I32;

    // Advanced options
    bool perFeaturePrecision = false;   ///< Allow different precision per feature
    u8   i16ThresholdBudgetPct = 10;    ///< Max % of nodes that may use i16 (if Auto)

    /// Check if this policy is pure int8-only (current behavior)
    /// Used to select int8-only fast path vs mixed-precision path
    bool is_int8_only() const noexcept {
        return threshold == Threshold::I8 &&
               residual == Residual::Q8_8 &&
               leaf == Leaf::I32 &&
               accum == Accum::I32 &&
               !perFeaturePrecision;
    }

    /// Fluent API for configuration
    PrecisionPolicy& set_threshold(Threshold t) noexcept {
        threshold = t;
        return *this;
    }

    PrecisionPolicy& set_residual(Residual r) noexcept {
        residual = r;
        return *this;
    }

    PrecisionPolicy& set_leaf(Leaf l) noexcept {
        leaf = l;
        return *this;
    }

    PrecisionPolicy& set_accum(Accum a) noexcept {
        accum = a;
        return *this;
    }

    PrecisionPolicy& set_i16_budget(u8 pct) noexcept {
        i16ThresholdBudgetPct = pct;
        return *this;
    }
};

} // namespace intgr
