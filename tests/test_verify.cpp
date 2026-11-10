// Verification against the host reference.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "bench/datagen.hpp"
#include "bench/verify.hpp"

namespace {

bench::Config make_config(std::size_t m, std::size_t n, std::size_t k, std::size_t pad) {
  bench::Config c;
  c.stage = "ref";
  c.ld_pad = pad;
  c.problem = {m, n, k, k + pad, n + pad, n + pad};
  return c;
}

// Correct C (padding NaN); reverse = sum in reverse k order with FMA (different rounding)
std::vector<double> product(const bench::Config& c, bool reverse = false) {
  const auto& p = c.problem;
  std::vector<double> out(p.m * p.ldc, NAN);
  for (std::size_t i = 0; i < p.m; ++i)
    for (std::size_t j = 0; j < p.n; ++j) {
      double s = 0.0;
      for (std::size_t t = 0; t < p.k; ++t) {
        const std::size_t kk = reverse ? p.k - 1 - t : t;
        const double a = bench::gen_value(c.seed, bench::MatrixId::A, i, kk);
        const double b = bench::gen_value(c.seed, bench::MatrixId::B, kk, j);
        s = reverse ? std::fma(a, b, s) : s + a * b;
      }
      out[i * p.ldc + j] = s;
    }
  return out;
}

TEST(Verify, CorrectResultPasses) {
  const auto c = make_config(57, 33, 100, 3);
  for (bool reverse : {false, true}) {
    const auto r = bench::verify(c, product(c, reverse).data());
    EXPECT_TRUE(r.performed);
    EXPECT_TRUE(r.passed) << r.failure;
    EXPECT_EQ(r.entries_checked, 57u * 33u);
  }
}

TEST(Verify, DetectsErrors) {
  const auto c = make_config(57, 33, 100, 3);
  const auto& p = c.problem;
  auto expect_fail = [&](std::vector<double> data, const char* what) {
    const auto r = bench::verify(c, data.data());
    EXPECT_FALSE(r.passed) << what;
    EXPECT_FALSE(r.failure.empty()) << what;
  };
  { auto d = product(c); d[10 * p.ldc + 5] *= 1.000001; expect_fail(d, "one element slightly off"); }
  { auto d = product(c); d[(p.m - 1) * p.ldc + p.n - 1] = NAN; expect_fail(d, "last element unwritten"); }
  { auto d = product(c); d[2 * p.ldc + p.n] = 0.0; expect_fail(d, "padding written"); }
  { auto d = product(c); for (std::size_t j = 0; j < p.n; ++j) d[(p.m - 1) * p.ldc + j] = 0.0; expect_fail(d, "last row missing"); }
}

TEST(Verify, Disabled) {
  auto c = make_config(4, 4, 4, 0);
  c.verify = false;
  const auto r = bench::verify(c, nullptr);
  EXPECT_FALSE(r.performed);
}

} // namespace
