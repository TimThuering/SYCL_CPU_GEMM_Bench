#pragma once
// Sanity check after initialization: a few sampled elements of A and B (incl. corners and
// padding) are read back and compared bit-exactly with the host value function,
// C is checked for the sentinel. Catches broken init kernels / address mix-ups early.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bench/config.hpp"
#include "bench/datagen.hpp"

namespace bench {

struct InitSample {
  enum class Matrix { A, B, C } matrix;
  std::size_t row, col;
  std::size_t offset;  // element offset within the matrix (row * ld + col)
};

std::vector<InitSample> make_init_samples(const Problem& p, std::uint64_t seed, std::size_t count);

// values[i] belongs to samples[i]. Throws std::runtime_error on the first mismatch.
void check_init_samples(const Problem& p, std::uint64_t seed,
                        const std::vector<InitSample>& samples, const std::vector<double>& values);

} // namespace bench
