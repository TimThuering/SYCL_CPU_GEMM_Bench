#pragma once
// Timing of the benchmark iterations with std::chrono::steady_clock (monotonic wall clock).
//
// One iteration = one call of run_once(), which must block until the computation is finished
// (SYCL: submit + wait). Sequence: 1 cold iteration, `warmup` warmup iterations, `reps` timed
// iterations. All durations are recorded; mean / min / max cover the timed iterations only.

#include <chrono>
#include <cstddef>
#include <iosfwd>
#include <vector>

namespace bench {

using bench_clock = std::chrono::steady_clock;
static_assert(bench_clock::is_steady, "the benchmark clock must be monotonic");

struct TimingResult {
  double cold_s = 0.0;               // first iteration (includes e.g. JIT compilation)
  std::vector<double> warmup_s;      // warmup iterations
  std::vector<double> times_s;       // timed iterations (raw data for post-processing)
  double mean_s = 0.0, min_s = 0.0, max_s = 0.0;  // over times_s
};

// Duration of one call of f in seconds
template <class F>
double time_once(F&& f) {
  const auto t0 = bench_clock::now();
  f();
  const auto t1 = bench_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

void compute_stats(TimingResult& t);

template <class F>
TimingResult time_iterations(F&& run_once, std::size_t warmup, std::size_t reps) {
  TimingResult t;
  t.cold_s = time_once(run_once);
  t.warmup_s.reserve(warmup);
  for (std::size_t i = 0; i < warmup; ++i) t.warmup_s.push_back(time_once(run_once));
  t.times_s.reserve(reps);
  for (std::size_t i = 0; i < reps; ++i) t.times_s.push_back(time_once(run_once));
  compute_stats(t);
  return t;
}

// GFLOP/s for a given operation count and duration
inline double gflops(double flops, double seconds) { return flops / seconds * 1e-9; }

void print_timing(std::ostream& os, const TimingResult& t, double flops);

} // namespace bench
