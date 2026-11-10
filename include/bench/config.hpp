#pragma once
// Run configuration: one run = one stage, one problem size, one result.

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>

namespace bench {

// Problem: C (M x N) = A (M x K) * B (K x N), row-major, double.
// Leading dimensions are in elements: lda >= K, ldb >= N, ldc >= N.
struct Problem {
  std::size_t m = 0, n = 0, k = 0;
  std::size_t lda = 0, ldb = 0, ldc = 0;

  double flops() const { return 2.0 * double(m) * double(n) * double(k); }
};

struct Config {
  // Kernel
  std::string stage;

  // Problem
  Problem problem;
  std::size_t ld_pad = 0;      // extra elements per row (applied to lda, ldb, ldc)
  std::uint64_t seed = 123;    // data generation seed

  // Run
  std::size_t warmup = 2;      // untimed iterations after the cold one
  std::size_t reps = 10;       // timed iterations
  bool profiling = false;      // additionally record SYCL event times

  // Verification
  bool verify = true;          // compare C with a host reference after the timed iterations

  // Output
  std::string output;          // JSON file path; empty = stdout
  std::string label;           // free-form tag for the scripts
  bool quiet = false;          // no human-readable summary on stderr

  std::string command_line;    // argv joined, recorded in the result
};

// Human-readable summary
void print_config(std::ostream& os, const Config& cfg);

} // namespace bench
