#include "bench/init_check.hpp"

#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace bench {

namespace {

const char* name(InitSample::Matrix m) {
  switch (m) {
    case InitSample::Matrix::A: return "A";
    case InitSample::Matrix::B: return "B";
    case InitSample::Matrix::C: return "C";
  }
  return "?";
}

void add(std::vector<InitSample>& s, InitSample::Matrix m, std::size_t row, std::size_t col, std::size_t ld) {
  s.push_back({m, row, col, row * ld + col});
}

// rows x ld region: four corners, last padding element (if any), plus `count` pseudo-random positions
void add_region(std::vector<InitSample>& s, InitSample::Matrix m, std::size_t rows, std::size_t cols,
                std::size_t ld, std::uint64_t& state, std::size_t count) {
  add(s, m, 0, 0, ld);
  add(s, m, 0, cols - 1, ld);
  add(s, m, rows - 1, 0, ld);
  add(s, m, rows - 1, cols - 1, ld);
  if (ld > cols) add(s, m, rows - 1, ld - 1, ld);
  for (std::size_t i = 0; i < count; ++i) {
    state = splitmix64(state);
    const std::size_t row = static_cast<std::size_t>(state % rows);
    state = splitmix64(state);
    const std::size_t col = static_cast<std::size_t>(state % ld);
    add(s, m, row, col, ld);
  }
}

bool same_bits(double x, double y) {
  return std::memcmp(&x, &y, sizeof(double)) == 0;
}

} // namespace

std::vector<InitSample> make_init_samples(const Problem& p, std::uint64_t seed, std::size_t count) {
  std::vector<InitSample> s;
  std::uint64_t state = seed ^ 0x5EED5EED5EED5EEDull;
  add_region(s, InitSample::Matrix::A, p.m, p.k, p.lda, state, count);
  add_region(s, InitSample::Matrix::B, p.k, p.n, p.ldb, state, count);
  add_region(s, InitSample::Matrix::C, p.m, p.n, p.ldc, state, count / 4);
  return s;
}

void check_init_samples(const Problem& p, std::uint64_t seed,
                        const std::vector<InitSample>& samples, const std::vector<double>& values) {
  if (samples.size() != values.size())
    throw std::logic_error("check_init_samples: size mismatch");

  for (std::size_t i = 0; i < samples.size(); ++i) {
    const InitSample& s = samples[i];
    const double got = values[i];
    double expected = std::nan("");
    bool ok = false;

    if (s.matrix == InitSample::Matrix::C) {
      ok = std::isnan(got);  // sentinel
    } else {
      const bool is_a = s.matrix == InitSample::Matrix::A;
      const std::size_t cols = is_a ? p.k : p.n;
      if (s.col >= cols) {
        ok = std::isnan(got);  // padding
      } else {
        expected = gen_value(seed, is_a ? MatrixId::A : MatrixId::B, s.row, s.col);
        ok = same_bits(got, expected);
      }
    }

    if (!ok) {
      std::ostringstream msg;
      msg.precision(17);
      msg << "init check failed: " << name(s.matrix) << "(" << s.row << ", " << s.col << ") = " << got
          << ", expected " << expected;
      throw std::runtime_error(msg.str());
    }
  }
}

} // namespace bench
