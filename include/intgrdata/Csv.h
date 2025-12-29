#pragma once

#include "../intgrcore/Types.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace intgr {

/// Simple CSV parser (no dependencies, handles basic cases)
/// Assumes: comma-separated, optional header, no quotes/escapes
class CsvReader {
public:
    std::vector<std::vector<std::string>> rows_;
    std::vector<std::string> header_;
    bool has_header_ = false;

    /// Load CSV from file
    /// @param path Path to CSV file
    /// @param has_header  True if first row is header
    /// @param max_rows  Maximum rows to read (0 = unlimited)
    /// @return Number of rows read (excluding header)
    usize load(const std::string& path, bool has_header = false, usize max_rows = 0) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open CSV file: " + path);
        }

        has_header_ = has_header;
        rows_.clear();
        header_.clear();

        std::string line;
        bool first_row = true;
        usize row_count = 0;

        while (std::getline(file, line) && (max_rows == 0 || row_count < max_rows)) {
            // Skip empty lines
            if (line.empty()) continue;

            // Remove trailing \r if present (Windows line endings)
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            std::vector<std::string> cols = parse_line(line);

            if (first_row && has_header) {
                header_ = cols;
                first_row = false;
                continue;
            }

            rows_.push_back(std::move(cols));
            row_count++;
            first_row = false;
        }

        return rows_.size();
    }

    /// Get number of rows (excluding header)
    usize num_rows() const noexcept {
        return rows_.size();
    }

    /// Get number of columns (from first row)
    usize num_cols() const noexcept {
        return rows_.empty() ? 0 : rows_[0].size();
    }

    /// Access raw row data
    const std::vector<std::string>& row(usize r) const {
        return rows_[r];
    }

    /// Access cell
    const std::string& cell(usize r, usize c) const {
        return rows_[r][c];
    }

    /// Parse column as floats
    /// @param col_idx  Column index
    /// @param out  Output buffer (must have space for num_rows() elements)
    /// @param nan_replacement  Value to use for parsing failures (default 0.0)
    void parse_column_float(usize col_idx, f32* out, f32 nan_replacement = 0.0f) const {
        for (usize r = 0; r < rows_.size(); ++r) {
            if (col_idx >= rows_[r].size()) {
                out[r] = nan_replacement;
                continue;
            }

            const std::string& val = rows_[r][col_idx];
            try {
                out[r] = std::stof(val);
            } catch (...) {
                out[r] = nan_replacement;
            }
        }
    }

    /// Parse column as integers
    void parse_column_int(usize col_idx, i32* out, i32 nan_replacement = 0) const {
        for (usize r = 0; r < rows_.size(); ++r) {
            if (col_idx >= rows_[r].size()) {
                out[r] = nan_replacement;
                continue;
            }

            const std::string& val = rows_[r][col_idx];
            try {
                out[r] = std::stoi(val);
            } catch (...) {
                out[r] = nan_replacement;
            }
        }
    }

    /// Get header names
    const std::vector<std::string>& header() const noexcept {
        return header_;
    }

private:
    /// Parse a single CSV line (comma-separated, no quote handling)
    std::vector<std::string> parse_line(const std::string& line) const {
        std::vector<std::string> cols;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            // Trim whitespace
            cell.erase(0, cell.find_first_not_of(" \t"));
            cell.erase(cell.find_last_not_of(" \t") + 1);
            cols.push_back(cell);
        }

        return cols;
    }
};

} // namespace intgr
