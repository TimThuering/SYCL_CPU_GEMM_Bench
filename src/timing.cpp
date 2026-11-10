#include "bench/timing.hpp"

#include <algorithm>
#include <numeric>
#include <ostream>

namespace bench {

void compute_stats(TimingResult& t) {
  if (t.times_s.empty()) return;
  const auto [mn, mx] = std::minmax_element(t.times_s.begin(), t.times_s.end());
  t.min_s = *mn;
  t.max_s = *mx;
  t.mean_s = std::accumulate(t.times_s.begin(), t.times_s.end(), 0.0) / static_cast<double>(t.times_s.size());
}

void print_timing(std::ostream& os, const TimingResult& t, double flops) {
  os << "time [ms]:     cold " << t.cold_s * 1e3 << ", timed (" << t.times_s.size() << ") mean "
     << t.mean_s * 1e3 << " min " << t.min_s * 1e3 << " max " << t.max_s * 1e3 << "\n"
     << "GFLOP/s:       mean " << gflops(flops, t.mean_s) << ", best " << gflops(flops, t.min_s) << "\n";
}

} // namespace bench
