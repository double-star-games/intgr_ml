#pragma once

#include "Types.h"

namespace intgr {

/// Utilities for file I/O operations (checksums, validation, etc.)
namespace FileUtils {

/// Compute simple checksum by summing all bytes (mod 2^32)
/// Used for basic integrity validation in binary file formats
/// @param data Pointer to data to checksum
/// @param size Number of bytes to checksum
/// @return Simple additive checksum value
inline u32 compute_checksum(const void* data, usize size) {
    const u8* bytes = static_cast<const u8*>(data);
    u32 sum = 0;
    for (usize i = 0; i < size; ++i) {
        sum += bytes[i];
    }
    return sum;
}

} // namespace FileUtils
} // namespace intgr
