#pragma once

#include "../intgrcore/Types.h"
#include "../intgrcore/MatrixInt.h"
#include "../intgrcore/Quantizer.h"
#include "Csv.h"
#include "LabelEncoder.h"
#include <vector>
#include <string>
#include <memory>

namespace intgr {

/// CSV loading quantization strategy
enum class CsvLoadStrategy {
    LINEAR,         ///< Simple linear quantization using fixed clip bounds (canonical, recommended)
    AUTO,           ///< Automatic int8/int16 selection based on precision threshold (legacy)
    HYBRID,         ///< Use feature importance to decide int8 vs int16 per column
    ADAPTIVE_BINS   ///< Allocate quantization bins proportionally to feature importance
};

/// Configuration for CSV loading
struct CsvLoadConfig {
    bool has_header = false;              ///< True if first row is header
    i32 label_col_idx = -1;               ///< Column index for labels (-1 = no labels)
    f32 clip_min = -10.0f;                ///< Minimum value for clipping
    f32 clip_max = 10.0f;                 ///< Maximum value for clipping
    f32 precision_threshold = 0.02f;      ///< Max quantization error for int8 (2%)
    CsvLoadStrategy strategy = CsvLoadStrategy::LINEAR;  ///< Quantization strategy (LINEAR is canonical)

    // Hybrid strategy parameters
    std::vector<f32> feature_importance;  ///< Importance scores [0, 1] per feature
    f32 importance_threshold = 0.1f;      ///< Features above this use int16 (top 10%)

    /// Create default configuration
    static CsvLoadConfig defaults() {
        return CsvLoadConfig{};
    }

    /// Create configuration for hybrid precision
    static CsvLoadConfig hybrid(const std::vector<f32>& importance, f32 threshold = 0.1f) {
        CsvLoadConfig cfg;
        cfg.strategy = CsvLoadStrategy::HYBRID;
        cfg.feature_importance = importance;
        cfg.importance_threshold = threshold;
        return cfg;
    }

    /// Create configuration for adaptive bins
    static CsvLoadConfig adaptive_bins(const std::vector<f32>& importance) {
        CsvLoadConfig cfg;
        cfg.strategy = CsvLoadStrategy::ADAPTIVE_BINS;
        cfg.feature_importance = importance;
        return cfg;
    }
};

/// Quantized dataset in Structure-of-Arrays (SoA) layout
/// Features stored column-wise as int8 or int16 for cache efficiency
class Dataset {
public:
    /// Column storage (each column is either int8 or int16)
    struct Column {
        std::vector<i8> data_i8;     ///< int8 data (if bits == 8)
        std::vector<i16> data_i16;   ///< int16 data (if bits == 16)
        i8 bits = 8;                 ///< Bit depth: 8 or 16

        /// Get pointer to int8 data (caller must verify bits == 8)
        const i8* data_as_i8() const noexcept {
            return data_i8.data();
        }

        /// Get pointer to int16 data (caller must verify bits == 16)
        const i16* data_as_i16() const noexcept {
            return data_i16.data();
        }

        /// Get value at index (handles both int8 and int16)
        i32 get(usize idx) const noexcept {
            return bits == 8 ? static_cast<i32>(data_i8[idx])
                             : static_cast<i32>(data_i16[idx]);
        }
    };

    std::vector<Column> columns_;       ///< Feature columns (SoA layout)
    std::vector<i32> labels_;           ///< Target labels (int32 for flexibility)
    Quantizer quantizer_;               ///< Quantization metadata
    LabelEncoder label_encoder_;        ///< Label encoder for text labels (optional)
    usize rows_ = 0;                    ///< Number of samples
    usize cols_ = 0;                    ///< Number of features

    Dataset() = default;

    /// Get dimensions
    usize rows() const noexcept { return rows_; }
    usize cols() const noexcept { return cols_; }

    /// Access column
    const Column& column(usize col_idx) const noexcept {
        return columns_[col_idx];
    }

    /// Get single feature value as int8 (for tree building)
    /// @param row_idx  Row index
    /// @param col_idx  Column index
    /// @return Feature value as int8 (downsampled if int16)
    i8 get_i8(usize row_idx, usize col_idx) const noexcept {
        const Column& col = columns_[col_idx];
        if (col.bits == 8) {
            return col.data_i8[row_idx];
        } else {
            // Downsample int16 to int8 (lossy)
            return static_cast<i8>(clamp<i32>(col.data_i16[row_idx], -128, 127));
        }
    }

    /// Access labels
    const i32* labels() const noexcept {
        return labels_.data();
    }

    i32* labels_mut() noexcept {
        return labels_.data();
    }

    /// Get quantizer
    const Quantizer& quantizer() const noexcept {
        return quantizer_;
    }

    /// Get label encoder (may not be fitted if labels were numeric)
    const LabelEncoder& label_encoder() const noexcept {
        return label_encoder_;
    }

    /// Check if labels were text (encoder was used)
    bool has_text_labels() const noexcept {
        return label_encoder_.is_fitted();
    }

    /// Load dataset from CSV file with hybrid precision based on feature importance
    /// @param path  Path to CSV file
    /// @param has_header  True if first row is header
    /// @param label_col_idx  Column index for labels (-1 = no labels)
    /// @param feature_importance  Importance scores [0, 1] per feature (empty = auto-detect)
    /// @param importance_threshold  Features above this importance use int16 (default 0.1 = top 10%)
    /// @param clip_min  Minimum value for clipping (default -10.0)
    /// @param clip_max  Maximum value for clipping (default +10.0)
    static Dataset from_csv_hybrid(const std::string& path,
                                   bool has_header,
                                   i32 label_col_idx,
                                   const std::vector<f32>& feature_importance,
                                   f32 importance_threshold = 0.1f,
                                   f32 clip_min = -10.0f,
                                   f32 clip_max = 10.0f);

    /// Load dataset from CSV file with adaptive bin allocation
    /// Allocates quantization bins proportionally to feature importance
    /// @param path  Path to CSV file
    /// @param has_header  True if first row is header
    /// @param label_col_idx  Column index for labels (-1 = no labels)
    /// @param feature_importance  Importance scores [0, 1] per feature
    /// @param clip_min  Minimum value for clipping (default -10.0)
    /// @param clip_max  Maximum value for clipping (default +10.0)
    /// @param precision_threshold  Max quantization error for int8 (default 0.02 = 2%)
    static Dataset from_csv_adaptive_bins(const std::string& path,
                                           bool has_header,
                                           i32 label_col_idx,
                                           const std::vector<f32>& feature_importance,
                                           f32 clip_min = -10.0f,
                                           f32 clip_max = 10.0f,
                                           f32 precision_threshold = 0.02f);

    /// Load dataset from CSV file with automatic quantization
    /// @param path  Path to CSV file
    /// @param has_header  True if first row is header
    /// @param label_col_idx  Column index for labels (-1 = no labels)
    /// @param clip_min  Minimum value for clipping (default -10.0)
    /// @param clip_max  Maximum value for clipping (default +10.0)
    /// @param precision_threshold  Max quantization error for int8 (default 0.02 = 2%)
    static Dataset from_csv(const std::string& path,
                            bool has_header = false,
                            i32 label_col_idx = -1,
                            f32 clip_min = -10.0f,
                            f32 clip_max = 10.0f,
                            f32 precision_threshold = 0.02f);

    /// Create a row-wise view (AoS) for a single sample
    /// Useful for inference on single examples
    /// @param row_idx  Row index
    /// @param out_i8  Output buffer for int8 features (size = cols)
    /// @param out_i16  Output buffer for int16 features (optional)
    /// @note This is slower than batch processing; prefer column-wise operations
    void copy_row_i8(usize row_idx, i8* out_i8) const {
        for (usize c = 0; c < cols_; ++c) {
            const Column& col = columns_[c];
            if (col.bits == 8) {
                out_i8[c] = col.data_i8[row_idx];
            } else {
                // Downsample int16 to int8 (lossy)
                out_i8[c] = static_cast<i8>(clamp<i32>(col.data_i16[row_idx], -128, 127));
            }
        }
    }

    /// Check if dataset has mixed int8/int16 columns
    bool has_mixed_precision() const noexcept {
        if (columns_.empty()) return false;

        i8 first_bits = columns_[0].bits;
        for (const auto& col : columns_) {
            if (col.bits != first_bits) return true;
        }
        return false;
    }

    /// Count number of int16 columns
    usize count_int16_columns() const noexcept {
        usize count = 0;
        for (const auto& col : columns_) {
            if (col.bits == 16) ++count;
        }
        return count;
    }

    /// Clone dataset with different labels (for OvR training)
    /// @param new_labels  New label array (must be same size as rows())
    /// @return Dataset with same features but different labels
    Dataset clone_with_labels(const i32* new_labels) const {
        Dataset cloned;
        cloned.rows_ = rows_;
        cloned.cols_ = cols_;
        cloned.quantizer_ = quantizer_;
        cloned.label_encoder_ = label_encoder_;

        // Deep copy of columns (vector copy constructor handles data)
        cloned.columns_ = columns_;

        // Copy new labels
        cloned.labels_.assign(new_labels, new_labels + rows_);

        return cloned;
    }

private:
    /// Unified CSV loading implementation (all strategies)
    /// Called by from_csv(), from_csv_hybrid(), and from_csv_adaptive_bins()
    static Dataset from_csv_impl(const std::string& path, const CsvLoadConfig& config);
};

} // namespace intgr
