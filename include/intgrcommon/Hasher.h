#pragma once

#include "../intgrcore/Types.h"
#include <string>
#include <vector>

namespace intgr {
namespace common {

/// SHA-256 hashing utility for determinism verification
/// Used to compute cryptographic hashes of:
/// - Model file bytes (for bit-exact model comparison)
/// - Logits buffers (for prediction comparison)
/// - CSV text (for output comparison)
class Hasher {
public:
    /// Compute SHA-256 of file contents
    /// @param path  Path to file
    /// @return Hex string (64 chars) or empty string on error
    static std::string digest_file(const std::string& path);

    /// Compute SHA-256 of byte buffer
    /// @param data  Raw bytes
    /// @param size  Buffer size in bytes
    /// @return Hex string (64 chars)
    static std::string digest_bytes(const u8* data, usize size);

    /// Compute SHA-256 of vector<i32> as raw bytes
    /// @param logits  Vector of int32 logits
    /// @return Hex string (64 chars)
    static std::string digest_logits(const std::vector<i32>& logits);

    /// Compute SHA-256 of vector<i32> as raw bytes (pointer version)
    /// @param logits  Pointer to int32 array
    /// @param count   Number of elements
    /// @return Hex string (64 chars)
    static std::string digest_logits(const i32* logits, usize count);

    /// Normalize CSV text (ensure \n line endings) and hash
    /// Reads file, replaces \r\n → \n, then hashes normalized text
    /// @param path  Path to CSV file
    /// @return Hex string (64 chars) or empty string on error
    static std::string digest_csv_text(const std::string& path);

private:
    /// Compute raw SHA-256 digest
    /// @param data    Input bytes
    /// @param size    Input size
    /// @param output  Output buffer (must be 32 bytes)
    static void sha256_raw(const u8* data, usize size, u8 output[32]);

    /// Convert raw bytes to hex string
    /// @param bytes  Raw bytes
    /// @param size   Number of bytes
    /// @return Hex string (lowercase)
    static std::string bytes_to_hex(const u8* bytes, usize size);

    // Prevent instantiation
    Hasher() = delete;
};

} // namespace common
} // namespace intgr
