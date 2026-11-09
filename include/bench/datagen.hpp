#pragma once
// Deterministic input data: value = f(seed, matrix, row, col).
//
// Counter-based (no RNG state), so every element can be generated independently:
// in parallel by the init kernel, and again on the host for verification.
// Only integer operations plus an exact scaling, so host and device produce
// bit-identical values. Usable in SYCL device code (no library calls).

#include <cstddef>
#include <cstdint>

namespace bench {

enum class MatrixId : std::uint64_t { A = 1, B = 2 };

// Stateless splitmix64: one step of the SplitMix64 generator applied to the input as state
// (Steele, Lea, Flood, OOPSLA 2014, https://doi.org/10.1145/2714064.2660195;
//  constants as in https://prng.di.unimi.it/splitmix64.c).
inline std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Uniform double in [-1, 1), 53 random bits.
// (h >> 11) * 2^-53 is exact; 2u - 1 is exact as well, so FMA contraction cannot change the result.
inline double gen_value(std::uint64_t seed, MatrixId id, std::size_t row, std::size_t col) {
  std::uint64_t h = splitmix64(seed ^ (static_cast<std::uint64_t>(id) << 56));
  h = splitmix64(h ^ static_cast<std::uint64_t>(row));
  h = splitmix64(h ^ static_cast<std::uint64_t>(col));
  const double u = static_cast<double>(h >> 11) * 0x1.0p-53;
  return 2.0 * u - 1.0;
}

} // namespace bench
