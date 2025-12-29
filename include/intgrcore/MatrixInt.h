#pragma once

#include "Types.h"
#include <cstring>
#include <vector>

namespace intgr {

/// Lightweight POD matrix wrapper for integer data
/// Row-major storage with optional stride support
/// No virtual methods, no heap allocation in hot paths
template<typename T>
struct Matrix {
    T* data_ = nullptr;
    usize rows_ = 0;
    usize cols_ = 0;
    usize stride_ = 0;  // Bytes between rows; defaults to cols_

    /// Create matrix view from existing buffer
    static constexpr Matrix make(T* ptr, usize rows, usize cols, usize stride = 0) noexcept {
        Matrix m;
        m.data_ = ptr;
        m.rows_ = rows;
        m.cols_ = cols;
        m.stride_ = stride ? stride : cols;
        return m;
    }

    /// Create matrix view from std::vector
    static Matrix from_vector(std::vector<T>& vec, usize rows, usize cols) noexcept {
        return make(vec.data(), rows, cols);
    }

    /// Access row pointer (mutable)
    inline T* row(usize r) noexcept {
        return data_ + r * stride_;
    }

    /// Access row pointer (const)
    inline const T* row(usize r) const noexcept {
        return data_ + r * stride_;
    }

    /// Element access (bounds not checked in release)
    inline T& operator()(usize r, usize c) noexcept {
        return row(r)[c];
    }

    /// Element access (const)
    inline const T& operator()(usize r, usize c) const noexcept {
        return row(r)[c];
    }

    /// Dimensions
    inline usize rows() const noexcept { return rows_; }
    inline usize cols() const noexcept { return cols_; }
    inline usize stride() const noexcept { return stride_; }

    /// Raw data pointer
    inline T* data() noexcept { return data_; }
    inline const T* data() const noexcept { return data_; }

    /// Size in elements (rows × cols, not accounting for stride)
    inline usize size() const noexcept { return rows_ * cols_; }

    /// Check if matrix is contiguous (stride == cols)
    inline bool is_contiguous() const noexcept { return stride_ == cols_; }

    /// Fill with value
    void fill(T value) noexcept {
        if (is_contiguous()) {
            std::fill_n(data_, rows_ * cols_, value);
        } else {
            for (usize r = 0; r < rows_; ++r) {
                std::fill_n(row(r), cols_, value);
            }
        }
    }

    /// Copy from another matrix (dimensions must match)
    void copy_from(const Matrix<T>& src) noexcept {
        if (is_contiguous() && src.is_contiguous() &&
            rows_ == src.rows_ && cols_ == src.cols_) {
            std::memcpy(data_, src.data_, rows_ * cols_ * sizeof(T));
        } else {
            for (usize r = 0; r < rows_; ++r) {
                std::memcpy(row(r), src.row(r), cols_ * sizeof(T));
            }
        }
    }
};

/// Convenience typedefs
using Mat8 = Matrix<i8>;
using Mat16 = Matrix<i16>;
using Mat32 = Matrix<i32>;
using Mat64 = Matrix<i64>;

/// Owning matrix (manages memory via std::vector)
template<typename T>
struct OwnedMatrix {
    std::vector<T> storage;
    usize rows_ = 0;
    usize cols_ = 0;

    OwnedMatrix() = default;

    OwnedMatrix(usize rows, usize cols)
        : storage(rows * cols), rows_(rows), cols_(cols) {}

    OwnedMatrix(usize rows, usize cols, T initial_value)
        : storage(rows * cols, initial_value), rows_(rows), cols_(cols) {}

    /// Get view of owned data
    Matrix<T> view() noexcept {
        return Matrix<T>::make(storage.data(), rows_, cols_);
    }

    /// Get const view
    Matrix<const T> view() const noexcept {
        return Matrix<const T>::make(storage.data(), rows_, cols_);
    }

    /// Element access
    inline T& operator()(usize r, usize c) noexcept {
        return storage[r * cols_ + c];
    }

    inline const T& operator()(usize r, usize c) const noexcept {
        return storage[r * cols_ + c];
    }

    usize rows() const noexcept { return rows_; }
    usize cols() const noexcept { return cols_; }
    usize size() const noexcept { return storage.size(); }

    void resize(usize rows, usize cols) {
        rows_ = rows;
        cols_ = cols;
        storage.resize(rows * cols);
    }

    void fill(T value) {
        std::fill(storage.begin(), storage.end(), value);
    }
};

} // namespace intgr
