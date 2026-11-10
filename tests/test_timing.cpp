// Timing loop and statistics.

#include <gtest/gtest.h>

#include "bench/timing.hpp"

namespace {

TEST(Timing, IterationCounts) {
  int calls = 0;
  const auto t = bench::time_iterations([&] { ++calls; }, 3, 5);
  EXPECT_EQ(calls, 1 + 3 + 5);
  EXPECT_EQ(t.warmup_s.size(), 3u);
  EXPECT_EQ(t.times_s.size(), 5u);
  EXPECT_GE(t.cold_s, 0.0);
}

TEST(Timing, NoWarmup) {
  const auto t = bench::time_iterations([] {}, 0, 1);
  EXPECT_TRUE(t.warmup_s.empty());
  EXPECT_EQ(t.times_s.size(), 1u);
}

TEST(Timing, Stats) {
  bench::TimingResult t;
  t.times_s = {0.3, 0.1, 0.2, 0.6};
  bench::compute_stats(t);
  EXPECT_DOUBLE_EQ(t.min_s, 0.1);
  EXPECT_DOUBLE_EQ(t.max_s, 0.6);
  EXPECT_DOUBLE_EQ(t.mean_s, 0.3);
}

TEST(Timing, Gflops) {
  EXPECT_DOUBLE_EQ(bench::gflops(2e9, 1.0), 2.0);
  EXPECT_DOUBLE_EQ(bench::gflops(2e9, 0.5), 4.0);
}

} // namespace
