#pragma once
// Verification of the computed C against a complete host reference (naive triple loop).
//
// A and B are not copied back: their values are regenerated with gen_value().
// Only C (M x ldc, incl. padding) is needed in host memory.
// Every element must match the reference within verify_tolerance (absolute difference),
// and the row padding of C must still hold the NaN sentinel.

#include <cstddef>
#include <iosfwd>
#include <string>

#include "bench/config.hpp"

namespace bench {

inline constexpr double verify_tolerance = 1e-8;

struct VerifyResult {
  bool performed = false;          // false if verification was disabled
  bool passed = false;
  std::size_t entries_checked = 0;
  double max_abs_error = 0.0;
  std::string failure;             // first failure, empty if passed
};

// c_host: C as M x ldc doubles (row-major, incl. padding)
VerifyResult verify(const Config& cfg, const double* c_host);

void print_verify_result(std::ostream& os, const VerifyResult& r);

} // namespace bench
