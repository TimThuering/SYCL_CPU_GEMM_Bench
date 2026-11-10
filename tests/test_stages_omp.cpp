// Every OpenMP twin stage on many problem shapes, verified against the host reference.
// Each stage runs twice: the second run starts from the result of the first, so a stage that
// accumulates onto C instead of overwriting it fails. Stages that are not implemented are skipped.

#include <gtest/gtest.h>

#include <vector>

#include "backend_omp/arena.hpp"
#include "backend_omp/stages.hpp"
#include "bench/errors.hpp"
#include "bench/make_args.hpp"
#include "bench/stages.hpp"
#include "bench/verify.hpp"
#include "stage_test_cases.hpp"

namespace {

using namespace stage_tests;

TEST(StageTable, AllStagesRegistered) {
  for (const auto& s : bench::stage_names) EXPECT_NE(bench::find_stage(s.name), nullptr) << s.name;
}

class OmpStageTest : public testing::TestWithParam<Param> {};

TEST_P(OmpStageTest, MatchesReference) {
  const auto& [name, shape] = GetParam();
  const bench::OmpStage* stage = bench::find_stage(name);
  ASSERT_NE(stage, nullptr);

  const bench::Config cfg = make_config(name, shape);
  bench::Arena arena(cfg.problem, stage->workspace_bytes(cfg.problem));
  arena.initialize(cfg.seed);
  const GemmArgs args = bench::make_gemm_args(arena.matrices(), cfg.problem);

  try {
    for (int run = 0; run < 2; ++run) stage->run(args);
  } catch (const bench::UnsupportedError& e) {
    GTEST_SKIP() << e.what();
  }

  std::vector<double> c(cfg.problem.m * cfg.problem.ldc);
  arena.copy_c_to_host(c.data());
  const bench::VerifyResult r = bench::verify(cfg, c.data());
  EXPECT_TRUE(r.passed) << r.failure;
}

INSTANTIATE_TEST_SUITE_P(Stages, OmpStageTest,
                         testing::Combine(testing::ValuesIn(stage_list()), testing::ValuesIn(shapes())),
                         param_name);

} // namespace
