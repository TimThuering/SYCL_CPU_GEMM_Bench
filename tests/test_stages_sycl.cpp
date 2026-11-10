// Every SYCL stage on many problem shapes, verified against the host reference.
// Each stage runs twice: the second run starts from the result of the first, so a stage that
// accumulates onto C instead of overwriting it fails. Stages that are not implemented are skipped.

#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

#include <vector>

#include "backend_sycl/arena.hpp"
#include "backend_sycl/stages.hpp"
#include "bench/errors.hpp"
#include "bench/make_args.hpp"
#include "bench/stages.hpp"
#include "bench/verify.hpp"
#include "stage_test_cases.hpp"

namespace {

using namespace stage_tests;

sycl::queue& test_queue() {
  static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
  return q;
}

TEST(StageTable, AllStagesRegistered) {
  for (const auto& s : bench::stage_names) EXPECT_NE(bench::find_stage(s.name), nullptr) << s.name;
}

class SyclStageTest : public testing::TestWithParam<Param> {};

TEST_P(SyclStageTest, MatchesReference) {
  const auto& [name, shape] = GetParam();
  const bench::SyclStage* stage = bench::find_stage(name);
  ASSERT_NE(stage, nullptr);

  const bench::Config cfg = make_config(name, shape);
  sycl::queue& q = test_queue();
  bench::Arena arena(q, cfg.problem, stage->workspace_bytes(cfg.problem));
  arena.initialize(cfg.seed);
  const GemmArgs args = bench::make_gemm_args(arena.matrices(), cfg.problem);

  try {
    for (int run = 0; run < 2; ++run) {
      stage->run(q, args);
      q.wait_and_throw();
    }
  } catch (const bench::UnsupportedError& e) {
    GTEST_SKIP() << e.what();
  }

  std::vector<double> c(cfg.problem.m * cfg.problem.ldc);
  arena.copy_c_to_host(c.data());
  const bench::VerifyResult r = bench::verify(cfg, c.data());
  EXPECT_TRUE(r.passed) << r.failure;
}

INSTANTIATE_TEST_SUITE_P(Stages, SyclStageTest,
                         testing::Combine(testing::ValuesIn(stage_list()), testing::ValuesIn(shapes())),
                         param_name);

} // namespace
