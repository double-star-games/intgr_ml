#pragma once

#include <string>
#include <variant>
#include <stdexcept>

namespace intgr {

/// Error codes for operations that can fail
enum class ErrorCode {
    None = 0,
    InvalidArgument,
    OutOfRange,
    IoError,
    ParseError,
    ModelError,
    RuntimeError
};

/// Error information for failed operations
struct Error {
    ErrorCode code = ErrorCode::None;
    std::string message;

    Error() = default;
    Error(ErrorCode c, const std::string& msg) : code(c), message(msg) {}
    Error(ErrorCode c, const char* msg) : code(c), message(msg) {}

    bool ok() const { return code == ErrorCode::None; }
    explicit operator bool() const { return !ok(); }  // true if error
};

/// Result type for operations that can fail
/// Usage:
///   Result<int> r = some_function();
///   if (r.ok()) { use(r.value()); }
///   else { handle(r.error()); }
template<typename T>
class Result {
public:
    /// Construct success result
    Result(const T& value) : data_(value) {}
    Result(T&& value) : data_(std::move(value)) {}

    /// Construct error result
    Result(const Error& err) : data_(err) {}
    Result(Error&& err) : data_(std::move(err)) {}
    Result(ErrorCode code, const std::string& msg) : data_(Error{code, msg}) {}
    Result(ErrorCode code, const char* msg) : data_(Error{code, msg}) {}

    /// Check if result is success
    bool ok() const { return std::holds_alternative<T>(data_); }
    explicit operator bool() const { return ok(); }

    /// Get value (throws if error)
    const T& value() const {
        if (!ok()) {
            throw std::runtime_error(error().message);
        }
        return std::get<T>(data_);
    }

    T& value() {
        if (!ok()) {
            throw std::runtime_error(error().message);
        }
        return std::get<T>(data_);
    }

    /// Get value or default
    T value_or(const T& default_value) const {
        return ok() ? std::get<T>(data_) : default_value;
    }

    /// Get error (only valid if !ok())
    const Error& error() const {
        return std::get<Error>(data_);
    }

private:
    std::variant<T, Error> data_;
};

/// Specialization for void results (just success/failure)
template<>
class Result<void> {
public:
    Result() : error_{} {}  // Success
    Result(const Error& err) : error_(err) {}
    Result(Error&& err) : error_(std::move(err)) {}
    Result(ErrorCode code, const std::string& msg) : error_{code, msg} {}
    Result(ErrorCode code, const char* msg) : error_{code, msg} {}

    bool ok() const { return error_.code == ErrorCode::None; }
    explicit operator bool() const { return ok(); }

    const Error& error() const { return error_; }

private:
    Error error_;
};

/// Helper to create success result
template<typename T>
Result<T> Ok(T&& value) {
    return Result<T>(std::forward<T>(value));
}

inline Result<void> Ok() {
    return Result<void>();
}

/// Helper to create error result
template<typename T = void>
Result<T> Err(ErrorCode code, const std::string& msg) {
    return Result<T>(code, msg);
}

template<typename T = void>
Result<T> Err(ErrorCode code, const char* msg) {
    return Result<T>(code, msg);
}

} // namespace intgr
