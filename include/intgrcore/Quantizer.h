#pragma once

#include "Types.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace intgr {

/// Quantization metadata for a single feature column
struct QuantMeta {
    f32 scale = 1.0f;      ///< Scaling factor: x_int = round(x_float / scale) + zero
    i32 zero = 0;          ///< Zero-point offset
    i8  bits = 8;          ///< Bit depth: 8 or 16
    f32 min_val = 0.0f;    ///< Observed min value in training data
    f32 max_val = 0.0f;    ///< Observed max value in training data

    /// Percentile bin edges (256 values for percentile quantization, empty for uniform)
    std::vector<f32> percentile_bins;

    /// Check if this uses percentile quantization
    bool is_percentile() const noexcept {
        return !percentile_bins.empty();
    }

    /// Check if this column requires int16 (based on range or precision loss)
    bool requires_int16() const noexcept {
        return bits == 16;
    }

    /// Quantize single float value to int8
    i8 quantize_i8(f32 x) const noexcept {
        if (is_percentile()) {
            // Binary search to find which bin this value falls into
            // bin_edges[0] = min, bin_edges[255] = max
            // Find largest i such that bin_edges[i] <= x
            auto it = std::upper_bound(percentile_bins.begin(), percentile_bins.end(), x);
            if (it == percentile_bins.begin()) {
                return -128;  // Below min
            }
            i64 bin = (it - percentile_bins.begin()) - 1;
            return static_cast<i8>(clamp<i64>(bin - constants::I8_OFFSET, -128, 127));
        } else {
            // Uniform quantization
            i32 q = static_cast<i32>(std::round(x / scale)) + zero;
            return static_cast<i8>(clamp<i32>(q, -128, 127));
        }
    }

    /// Quantize single float value to int16
    i16 quantize_i16(f32 x) const noexcept {
        i32 q = static_cast<i32>(std::round(x / scale)) + zero;
        return static_cast<i16>(clamp<i32>(q, -32768, 32767));
    }

    /// Dequantize int8 back to float (for validation)
    f32 dequantize_i8(i8 v) const noexcept {
        // Handle constant column case
        if (min_val == max_val) {
            return min_val;
        }

        if (is_percentile()) {
            // Return the bin edge value (or midpoint of bin range)
            // bin_idx is always in [0, 255] after offset, safe to use as usize
            i32 raw_idx = static_cast<i32>(v) + constants::I8_OFFSET;
            usize bin_idx = (raw_idx < 0) ? 0 :
                            (static_cast<usize>(raw_idx) >= percentile_bins.size()) ?
                            percentile_bins.size() - 1 : static_cast<usize>(raw_idx);
            return percentile_bins[bin_idx];
        } else {
            // Uniform dequantization
            return static_cast<f32>((static_cast<i32>(v) - zero)) * scale;
        }
    }

    /// Dequantize int16 back to float (for validation)
    f32 dequantize_i16(i16 v) const noexcept {
        // Handle constant column case
        if (min_val == max_val) {
            return min_val;
        }
        return static_cast<f32>((static_cast<i32>(v) - zero)) * scale;
    }
};

/// Uniform quantizer for dataset features
/// Supports per-column int8 or int16 quantization with automatic promotion
class Quantizer {
public:
    std::vector<QuantMeta> columns_;  ///< Per-column quantization parameters

    Quantizer() = default;

    /// Reserve space for N columns
    void reserve(usize n_cols) {
        columns_.reserve(n_cols);
    }

    /// Get number of columns
    usize size() const noexcept {
        return columns_.size();
    }

    /// Access column metadata
    const QuantMeta& operator[](usize col) const noexcept {
        return columns_[col];
    }

    QuantMeta& operator[](usize col) noexcept {
        return columns_[col];
    }

    /// Fit quantization parameters for a single column
    /// @param data   Float array of length n
    /// @param n      Number of samples
    /// @param clip_min  Clip values below this (e.g., -10.0)
    /// @param clip_max  Clip values above this (e.g., +10.0)
    /// @param precision_threshold  Max acceptable quantization error (default 0.02 = 2%)
    /// @return Column index
    usize fit_column(const f32* data, usize n,
                     f32 clip_min = -10.0f, f32 clip_max = 10.0f,
                     f32 precision_threshold = 0.02f) {
        QuantMeta meta;

        // Find actual range
        f32 min_val = std::numeric_limits<f32>::max();
        f32 max_val = std::numeric_limits<f32>::lowest();

        for (usize i = 0; i < n; ++i) {
            f32 val = clamp(data[i], clip_min, clip_max);
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        // Handle constant column
        if (max_val - min_val < 1e-6f) {
            meta.min_val = min_val;
            meta.max_val = max_val;
            meta.scale = 1.0f;
            meta.zero = static_cast<i32>(std::round(min_val));  // Store the constant value
            meta.bits = 8;
            columns_.push_back(meta);
            return columns_.size() - 1;
        }

        // Auto-select quantization method based on skewness
        // For heavily skewed distributions, percentile quantization works better
        f32 skewness = compute_skewness(data, n);
        if (skewness > 1.0f || skewness < -1.0f) {
            // Highly skewed - use percentile quantization
            return fit_column_percentile(data, n, clip_min, clip_max);
        }

        // Use uniform quantization for symmetric/normal distributions
        // Try int8 first
        // Formula: x_quant = round((x - min_val) / scale) - I8_OFFSET
        // Equivalently: x_quant = round(x / scale) + zero_point
        // where zero_point = -128 - round(min_val / scale)
        f32 range = max_val - min_val;
        meta.scale = range / 255.0f;  // Map to [-128, 127] range
        meta.zero = -128 - static_cast<i32>(std::round(min_val / meta.scale));
        meta.bits = 8;
        meta.min_val = min_val;
        meta.max_val = max_val;

        // Check if int8 precision is acceptable
        // Compute max quantization error
        f32 max_error = 0.0f;
        for (usize i = 0; i < std::min(n, usize(1000)); ++i) {  // Sample first 1000 points
            f32 original = clamp(data[i], clip_min, clip_max);
            i8 quantized = meta.quantize_i8(original);
            f32 reconstructed = meta.dequantize_i8(quantized);
            f32 error = std::abs(reconstructed - original);
            f32 relative_error = error / (std::abs(original) + 1e-6f);
            max_error = std::max(max_error, relative_error);
        }

        // If int8 precision is insufficient, promote to int16
        if (max_error > precision_threshold) {
            meta.scale = range / 65535.0f;  // Map to [-32768, 32767] range
            meta.zero = -32768 - static_cast<i32>(std::round(min_val / meta.scale));
            meta.bits = 16;
        }

        columns_.push_back(meta);
        return columns_.size() - 1;
    }

    /// Fit simple linear quantization using fixed clip bounds (canonical method)
    /// This is the recommended quantization for most use cases.
    /// Maps [clip_min, clip_max] → [0, 255] → [-128, 127]
    /// Matches Python bindings behavior exactly.
    /// @param clip_min  Clip values below this (e.g., -10.0)
    /// @param clip_max  Clip values above this (e.g., +10.0)
    /// @return Column index
    usize fit_column_linear(f32 clip_min = -10.0f, f32 clip_max = 10.0f) {
        QuantMeta meta;
        meta.min_val = clip_min;
        meta.max_val = clip_max;
        meta.scale = (clip_max - clip_min) / 255.0f;
        meta.zero = -128 - static_cast<i32>(std::round(clip_min / meta.scale));
        meta.bits = 8;
        columns_.push_back(meta);
        return columns_.size() - 1;
    }

    /// Quantize a single value using linear quantization (canonical method)
    /// Clips to [clip_min, clip_max], then maps to [-128, 127]
    static i8 quantize_linear(f32 val, f32 clip_min, f32 clip_max) noexcept {
        val = clamp(val, clip_min, clip_max);
        f32 normalized = (val - clip_min) / (clip_max - clip_min);
        i32 quantized = static_cast<i32>(normalized * 255.0f);
        return static_cast<i8>(quantized - 128);
    }

    /// Fit percentile-based quantization for a single column with custom bin count
    /// Uses equal-frequency bins instead of equal-width bins
    /// @param data   Float array of length n
    /// @param n      Number of samples
    /// @param bin_count  Number of bins to use (64-512 range)
    /// @param clip_min  Clip values below this
    /// @param clip_max  Clip values above this
    /// @return Column index
    usize fit_column_percentile_adaptive(const f32* data, usize n, usize bin_count,
                                          f32 clip_min = -10.0f, f32 clip_max = 10.0f) {
        QuantMeta meta;

        // Handle empty data
        if (n == 0) {
            columns_.push_back(meta);
            return columns_.size() - 1;
        }

        // Clamp bin count to reasonable range
        bin_count = clamp<usize>(bin_count, 64, 512);

        // Copy and clip data
        std::vector<f32> sorted_data;
        sorted_data.reserve(n);
        for (usize i = 0; i < n; ++i) {
            sorted_data.push_back(clamp(data[i], clip_min, clip_max));
        }

        // Find min/max
        f32 min_val = *std::min_element(sorted_data.begin(), sorted_data.end());
        f32 max_val = *std::max_element(sorted_data.begin(), sorted_data.end());

        meta.min_val = min_val;
        meta.max_val = max_val;
        meta.bits = 8;

        // Handle constant column
        if (max_val - min_val < 1e-6f) {
            meta.scale = 1.0f;
            meta.zero = static_cast<i32>(std::round(min_val));
            columns_.push_back(meta);
            return columns_.size() - 1;
        }

        // Sort data to compute percentiles
        std::sort(sorted_data.begin(), sorted_data.end());

        // Compute N quantile bin edges (equal frequency bins)
        meta.percentile_bins.resize(bin_count);

        for (usize i = 0; i < bin_count; ++i) {
            // Compute the index in sorted data for this quantile
            usize idx = (i * n) / bin_count;
            if (idx >= n) idx = n - 1;
            meta.percentile_bins[i] = sorted_data[idx];
        }

        // Ensure bin edges are strictly increasing
        for (usize i = 1; i < bin_count; ++i) {
            if (meta.percentile_bins[i] <= meta.percentile_bins[i-1]) {
                meta.percentile_bins[i] = meta.percentile_bins[i-1] + 1e-6f;
            }
        }

        // Scale percentile bins back to 256 entries for int8 mapping
        // We need to interpolate from bin_count bins to 256 bins
        std::vector<f32> scaled_bins(256);
        for (usize i = 0; i < 256; ++i) {
            // Map from 256 bins to bin_count bins
            f32 src_idx = (static_cast<f32>(i) * static_cast<f32>(bin_count)) / 256.0f;
            usize idx_low = static_cast<usize>(src_idx);
            usize idx_high = std::min(idx_low + 1, bin_count - 1);
            f32 fraction = src_idx - static_cast<f32>(idx_low);

            // Linear interpolation between bins
            scaled_bins[i] = meta.percentile_bins[idx_low] * (1.0f - fraction) +
                             meta.percentile_bins[idx_high] * fraction;
        }
        meta.percentile_bins = scaled_bins;

        columns_.push_back(meta);
        return columns_.size() - 1;
    }

    /// Fit percentile-based quantization for a single column
    /// Uses equal-frequency bins instead of equal-width bins
    /// @param data   Float array of length n
    /// @param n      Number of samples
    /// @param clip_min  Clip values below this
    /// @param clip_max  Clip values above this
    /// @return Column index
    usize fit_column_percentile(const f32* data, usize n,
                                 f32 clip_min = -10.0f, f32 clip_max = 10.0f) {
        QuantMeta meta;

        // Handle empty data
        if (n == 0) {
            columns_.push_back(meta);
            return columns_.size() - 1;
        }

        // Copy and clip data
        std::vector<f32> sorted_data;
        sorted_data.reserve(n);
        for (usize i = 0; i < n; ++i) {
            sorted_data.push_back(clamp(data[i], clip_min, clip_max));
        }

        // Find min/max
        f32 min_val = *std::min_element(sorted_data.begin(), sorted_data.end());
        f32 max_val = *std::max_element(sorted_data.begin(), sorted_data.end());

        meta.min_val = min_val;
        meta.max_val = max_val;
        meta.bits = 8;

        // Handle constant column
        if (max_val - min_val < 1e-6f) {
            meta.scale = 1.0f;
            meta.zero = static_cast<i32>(std::round(min_val));
            columns_.push_back(meta);
            return columns_.size() - 1;
        }

        // Sort data to compute percentiles
        std::sort(sorted_data.begin(), sorted_data.end());

        // Compute 256 quantile bin edges (equal frequency bins)
        // Each bin should contain approximately n/256 samples
        meta.percentile_bins.resize(256);

        for (usize i = 0; i < 256; ++i) {
            // Compute the index in sorted data for this quantile
            // For bin i, we want the value at position (i * n) / 256
            usize idx = (i * n) / 256;
            if (idx >= n) idx = n - 1;
            meta.percentile_bins[i] = sorted_data[idx];
        }

        // Ensure bin edges are strictly increasing by removing duplicates
        // If we have many duplicate values, some bins will be unused
        for (usize i = 1; i < 256; ++i) {
            if (meta.percentile_bins[i] <= meta.percentile_bins[i-1]) {
                // Ensure strictly increasing by adding a small epsilon
                meta.percentile_bins[i] = meta.percentile_bins[i-1] + 1e-6f;
            }
        }

        columns_.push_back(meta);
        return columns_.size() - 1;
    }

    /// Quantize an entire column to int8
    /// @param src  Source float data
    /// @param n    Number of elements
    /// @param col_idx  Column index (must be fit already)
    /// @param dst  Destination int8 buffer (must have space for n elements)
    void quantize_column_i8(const f32* src, usize n, usize col_idx, i8* dst) const {
        const QuantMeta& meta = columns_[col_idx];
        for (usize i = 0; i < n; ++i) {
            dst[i] = meta.quantize_i8(src[i]);
        }
    }

    /// Quantize an entire column to int16
    void quantize_column_i16(const f32* src, usize n, usize col_idx, i16* dst) const {
        const QuantMeta& meta = columns_[col_idx];
        for (usize i = 0; i < n; ++i) {
            dst[i] = meta.quantize_i16(src[i]);
        }
    }

    /// Check if any column requires int16
    bool has_int16_columns() const noexcept {
        for (const auto& meta : columns_) {
            if (meta.requires_int16()) return true;
        }
        return false;
    }

    /// Get number of int16 columns
    usize count_int16_columns() const noexcept {
        usize count = 0;
        for (const auto& meta : columns_) {
            if (meta.requires_int16()) ++count;
        }
        return count;
    }

private:
    /// Compute skewness: measure of distribution asymmetry
    /// Skewness = 0: symmetric (uniform, normal)
    /// Skewness > 1: right-skewed (long tail on right)
    /// Skewness < -1: left-skewed (long tail on left)
    /// @return Skewness value (0 if data is empty or constant)
    static f32 compute_skewness(const f32* data, usize n) noexcept {
        if (n == 0) return 0.0f;

        // Compute mean
        f32 mean = 0.0f;
        for (usize i = 0; i < n; ++i) {
            mean += data[i];
        }
        mean /= static_cast<f32>(n);

        // Compute variance and third moment
        f32 variance = 0.0f;
        f32 third_moment = 0.0f;

        for (usize i = 0; i < n; ++i) {
            f32 diff = data[i] - mean;
            variance += diff * diff;
            third_moment += diff * diff * diff;
        }

        variance /= static_cast<f32>(n);
        third_moment /= static_cast<f32>(n);

        if (variance < 1e-6f) return 0.0f;  // Constant data

        f32 std_dev = std::sqrt(variance);
        return third_moment / (std_dev * std_dev * std_dev);
    }
};

} // namespace intgr
