// Every OpenCL reference stage on many problem shapes, verified against the host reference.
// Each stage runs twice: the second run starts from the result of the first, so a stage that
// accumulates onto C instead of overwriting it fails.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "backend_opencl/arena.hpp"
#include "backend_opencl/gemm_args.hpp"
#include "backend_opencl/program.hpp"
#include "backend_opencl/runtime.hpp"
#include "backend_opencl/stages.hpp"
#include "bench/verify.hpp"
#include "stage_test_cases.hpp"

namespace {

using namespace stage_tests;

TEST(StageTable, AllStagesRegistered) {
  for (const auto& s : bench::stage_names) EXPECT_NE(bench::find_stage(s.name), nullptr) << s.name;
}

class OpenClStageTest : public testing::TestWithParam<Param> {};

TEST_P(OpenClStageTest, MatchesReference) {
  const auto& [name, shape] = GetParam();
  const bench::ClStage* stage = bench::find_stage(name);
  ASSERT_NE(stage, nullptr);

  const bench::ClRuntime rt = bench::make_runtime();
  const bench::Config cfg = make_config(name, shape);
  bench::Arena arena(rt, cfg.problem, stage->workspace_bytes(cfg.problem));
  arena.initialize(cfg.seed);
  const bench::ClGemmArgs args = bench::make_cl_gemm_args(arena.matrices(), cfg.problem);

  bench::ClStageProgram program(rt, stage->source);
  for (int run = 0; run < 2; ++run) {
    stage->run(program, args);
    ASSERT_NO_THROW(bench::finish(rt));
  }

  std::vector<double> c(cfg.problem.m * cfg.problem.ldc);
  arena.copy_c_to_host(c.data());
  const bench::VerifyResult r = bench::verify(cfg, c.data());
  EXPECT_TRUE(r.passed) << r.failure;
}

INSTANTIATE_TEST_SUITE_P(Stages, OpenClStageTest,
                         testing::Combine(testing::ValuesIn(stage_list()), testing::ValuesIn(shapes())),
                         param_name);

TEST(StageProgram, BuildOptionsCarryKernelParameters) {
  EXPECT_EQ(bench::stage_build_options(), "-DMR=" + std::to_string(GEMM_MR) + " -DNR=" +
                                              std::to_string(GEMM_NR) + " -DKC=" + std::to_string(GEMM_KC) +
                                              " -DMC=" + std::to_string(GEMM_MC));
}

// Handles returned by kernel() stay valid while more kernels are added (the owners are moved
// when the internal list grows); a second lookup returns the same handle, no new kernel object.
TEST(StageProgram, KernelHandlesStayValid) {
  const bench::ClRuntime rt = bench::make_runtime();
  std::string source;
  for (int i = 0; i < 10; ++i) source += "__kernel void k" + std::to_string(i) + "(__global int* p) { p[0] = " + std::to_string(i) + "; }\n";
  bench::ClStageProgram program(rt, source.c_str());

  std::vector<cl_kernel> handles;
  for (int i = 0; i < 10; ++i) handles.push_back(program.kernel(("k" + std::to_string(i)).c_str()));

  for (int i = 0; i < 10; ++i) {
    const std::string name = "k" + std::to_string(i);
    EXPECT_EQ(program.kernel(name.c_str()), handles[i]) << name;
    char fn[16] = {};
    ASSERT_EQ(clGetKernelInfo(handles[i], CL_KERNEL_FUNCTION_NAME, sizeof(fn), fn, nullptr), CL_SUCCESS) << name;
    EXPECT_EQ(std::string(fn), name);
  }
}

TEST(StageProgram, BuildErrorReportsLog) {
  const bench::ClRuntime rt = bench::make_runtime();
  bench::ClStageProgram program(rt, "__kernel void broken( { }");
  try {
    program.kernel("broken");
    FAIL() << "expected a build error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("clBuildProgram failed"), std::string::npos) << e.what();
  }
}

} // namespace
