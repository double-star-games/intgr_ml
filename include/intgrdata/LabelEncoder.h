#pragma once

#include "../intgrcore/Types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace intgr {

/// Label encoder for converting text labels to integers
/// Useful for classification tasks where labels are strings (e.g., "yes"/"no", "<=50K"/">50K")
class LabelEncoder {
public:
    std::unordered_map<std::string, i32> label_to_int_;  ///< Label string → integer mapping
    std::vector<std::string> int_to_label_;              ///< Integer → label string mapping
    i32 num_classes_ = 0;                                ///< Number of unique classes

    LabelEncoder() = default;

    /// Fit encoder on array of label strings
    /// @param labels  Array of label strings
    /// @param count  Number of labels
    /// @return Number of unique classes discovered
    i32 fit(const std::string* labels, usize count) {
        label_to_int_.clear();
        int_to_label_.clear();
        num_classes_ = 0;

        // Build mapping from unique strings
        for (usize i = 0; i < count; ++i) {
            const std::string& label = labels[i];

            // Skip empty labels
            if (label.empty()) continue;

            // Add new label if not seen before
            if (label_to_int_.find(label) == label_to_int_.end()) {
                label_to_int_[label] = num_classes_;
                int_to_label_.push_back(label);
                num_classes_++;
            }
        }

        return num_classes_;
    }

    /// Transform label strings to integers
    /// @param labels  Array of label strings
    /// @param count  Number of labels
    /// @param out  Output array of integers (must have space for count elements)
    /// @param unknown_label  Value to use for unknown labels (default: -1)
    void transform(const std::string* labels, usize count, i32* out, i32 unknown_label = -1) const {
        for (usize i = 0; i < count; ++i) {
            const std::string& label = labels[i];
            auto it = label_to_int_.find(label);
            if (it != label_to_int_.end()) {
                out[i] = it->second;
            } else {
                out[i] = unknown_label;
            }
        }
    }

    /// Fit and transform in one call
    /// @param labels  Array of label strings
    /// @param count  Number of labels
    /// @param out  Output array of integers
    /// @return Number of unique classes
    i32 fit_transform(const std::string* labels, usize count, i32* out) {
        fit(labels, count);
        transform(labels, count, out);
        return num_classes_;
    }

    /// Inverse transform: convert integer back to label string
    /// @param encoded_label  Integer label
    /// @return Original label string
    /// @throws std::out_of_range if encoded_label is invalid
    const std::string& inverse_transform(i32 encoded_label) const {
        if (encoded_label < 0 || encoded_label >= num_classes_) {
            throw std::out_of_range("Invalid encoded label: " + std::to_string(encoded_label));
        }
        return int_to_label_[static_cast<usize>(encoded_label)];
    }

    /// Get number of unique classes
    i32 num_classes() const noexcept {
        return num_classes_;
    }

    /// Check if encoder has been fitted
    bool is_fitted() const noexcept {
        return num_classes_ > 0;
    }

    /// Get all unique label strings in order
    const std::vector<std::string>& classes() const noexcept {
        return int_to_label_;
    }

    /// Check if a label string is known
    bool contains(const std::string& label) const noexcept {
        return label_to_int_.find(label) != label_to_int_.end();
    }

    /// Get integer encoding for a label string
    /// @param label  Label string
    /// @param unknown_value  Value to return if label is unknown (default: -1)
    /// @return Integer encoding, or unknown_value if not found
    i32 encode(const std::string& label, i32 unknown_value = -1) const noexcept {
        auto it = label_to_int_.find(label);
        return (it != label_to_int_.end()) ? it->second : unknown_value;
    }

    /// Check if labels appear to be numeric
    /// @param labels  Array of label strings
    /// @param count  Number of labels to check
    /// @return True if all non-empty labels parse as integers
    static bool is_numeric(const std::string* labels, usize count) {
        for (usize i = 0; i < count; ++i) {
            const std::string& label = labels[i];
            if (label.empty()) continue;

            try {
                (void)std::stoi(label);  // Check if numeric, discard result
            } catch (...) {
                // Found non-numeric label
                return false;
            }
        }
        return true;
    }
};

} // namespace intgr
