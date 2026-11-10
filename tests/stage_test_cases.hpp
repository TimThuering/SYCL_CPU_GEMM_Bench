#pragma once
// Test cases for the stage tests (shared by the SYCL and the OpenMP twin tests):
// every stage x problem shapes (odd, rectangular, degenerate, with and without row padding).

#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <vector>

#include "bench/config.hpp"
#include "bench/stages.hpp"

namespace stage_tests {

struct Shape {
  std::size_t m, n, k, pad;
};

inline std::vector<Shape> shapes() {
  const std::vector<Shape> base = {
      {1, 1, 1, 0},    {7, 5, 3, 0},     {3, 7, 5, 0},      {33, 17, 65, 0},
      {64, 64, 64, 0}, {100, 1, 100, 0}, {1, 100, 100, 0},  {129, 131, 127, 0},
  };
  std::vector<Shape> out;
  for (const auto& s : base) {
    out.push_back(s);
    out.push_back({s.m, s.n, s.k, 3});
  }
  return out;
}

inline std::vector<std::string> stage_list() {
  std::vector<std::string> out;
  for (const auto& s : bench::stage_names) out.emplace_back(s.name);
  return out;
}

using Param = std::tuple<std::string, Shape>;

inline std::string param_name(const testing::TestParamInfo<Param>& info) {
  const auto& [stage, s] = info.param;
  return stage + "_" + std::to_string(s.m) + "x" + std::to_string(s.n) + "x" + std::to_string(s.k) +
         "_pad" + std::to_string(s.pad);
}

inline bench::Config make_config(const std::string& stage, const Shape& s) {
  bench::Config c;
  c.stage = stage;
  c.ld_pad = s.pad;
  c.problem = {s.m, s.n, s.k, s.k + s.pad, s.n + s.pad, s.n + s.pad};
  return c;
}

} // namespace stage_tests
