// Value function, arena layout, init sample check.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "bench/datagen.hpp"
#include "bench/init_check.hpp"
#include "bench/layout.hpp"

namespace {

using bench::MatrixId;

TEST(Datagen, DeterministicAndInRange) {
  for (std::size_t i = 0; i < 100; ++i)
    for (std::size_t j = 0; j < 100; ++j) {
      const double v = bench::gen_value(123, MatrixId::A, i, j);
      EXPECT_EQ(v, bench::gen_value(123, MatrixId::A, i, j));
      EXPECT_GE(v, -1.0);
      EXPECT_LT(v, 1.0);
    }
}

TEST(Datagen, DependsOnAllInputs) {
  const double v = bench::gen_value(123, MatrixId::A, 3, 4);
  EXPECT_NE(v, bench::gen_value(124, MatrixId::A, 3, 4));  // seed
  EXPECT_NE(v, bench::gen_value(123, MatrixId::B, 3, 4));  // matrix
  EXPECT_NE(v, bench::gen_value(123, MatrixId::A, 4, 3));  // row / column swapped
}

TEST(Datagen, MeanNearZero) {
  double sum = 0.0;
  const std::size_t n = 100000;
  for (std::size_t i = 0; i < n; ++i) sum += bench::gen_value(1, MatrixId::B, i / 1000, i % 1000);
  EXPECT_NEAR(sum / n, 0.0, 0.01);
}

TEST(Layout, RegionsPageAlignedAndDisjoint) {
  const bench::Problem p{513, 77, 257, 257 + 8, 77 + 8, 77 + 8};
  const auto l = bench::compute_layout(p, 4096 + 8);
  EXPECT_EQ(l.a_bytes, 513u * 265u * 8u);
  EXPECT_EQ(l.b_bytes, 257u * 85u * 8u);
  EXPECT_EQ(l.c_bytes, 513u * 85u * 8u);
  for (std::size_t off : {l.a_offset, l.b_offset, l.c_offset, l.ws_offset, l.total_bytes})
    EXPECT_EQ(off % bench::arena_page, 0u);
  EXPECT_GE(l.b_offset, l.a_offset + l.a_bytes);
  EXPECT_GE(l.c_offset, l.b_offset + l.b_bytes);
  EXPECT_GE(l.ws_offset, l.c_offset + l.c_bytes);
  EXPECT_GE(l.total_bytes, l.ws_offset + l.ws_bytes);
}

TEST(Layout, AlignmentCheck) {
  alignas(4096) static unsigned char buf[8192];
  EXPECT_NO_THROW(bench::check_arena_alignment(buf, "test"));
  EXPECT_THROW(bench::check_arena_alignment(buf + 16, "test"), std::runtime_error);
}

// Host-side copy of the arena initialization (A, B values with NaN padding, C NaN)
struct HostMatrices {
  std::vector<double> a, b, c;
  HostMatrices(const bench::Problem& p, std::uint64_t seed)
      : a(p.m * p.lda), b(p.k * p.ldb), c(p.m * p.ldc, NAN) {
    for (std::size_t i = 0; i < p.m; ++i)
      for (std::size_t j = 0; j < p.lda; ++j)
        a[i * p.lda + j] = j < p.k ? bench::gen_value(seed, MatrixId::A, i, j) : NAN;
    for (std::size_t i = 0; i < p.k; ++i)
      for (std::size_t j = 0; j < p.ldb; ++j)
        b[i * p.ldb + j] = j < p.n ? bench::gen_value(seed, MatrixId::B, i, j) : NAN;
  }
  std::vector<double> read(const std::vector<bench::InitSample>& samples) const {
    std::vector<double> v;
    for (const auto& s : samples)
      v.push_back((s.matrix == bench::InitSample::Matrix::A ? a : s.matrix == bench::InitSample::Matrix::B ? b : c)[s.offset]);
    return v;
  }
};

TEST(InitCheck, PassesOnCorrectData) {
  const bench::Problem p{37, 21, 50, 53, 24, 24};
  const HostMatrices h(p, 123);
  const auto samples = bench::make_init_samples(p, 123, 32);
  EXPECT_NO_THROW(bench::check_init_samples(p, 123, samples, h.read(samples)));
}

TEST(InitCheck, DetectsCorruption) {
  const bench::Problem p{37, 21, 50, 53, 24, 24};
  const auto samples = bench::make_init_samples(p, 123, 32);
  {
    HostMatrices h(p, 123);
    h.a[0] = 0.5;  // corner of A
    EXPECT_THROW(bench::check_init_samples(p, 123, samples, h.read(samples)), std::runtime_error);
  }
  {
    HostMatrices h(p, 123);
    h.b[(p.k - 1) * p.ldb + p.ldb - 1] = 0.0;  // padding of B
    EXPECT_THROW(bench::check_init_samples(p, 123, samples, h.read(samples)), std::runtime_error);
  }
  {
    HostMatrices h(p, 123);
    h.c[0] = 0.0;  // C sentinel
    EXPECT_THROW(bench::check_init_samples(p, 123, samples, h.read(samples)), std::runtime_error);
  }
  {
    const HostMatrices h(p, 124);  // other seed
    EXPECT_THROW(bench::check_init_samples(p, 123, samples, h.read(samples)), std::runtime_error);
  }
}

} // namespace
