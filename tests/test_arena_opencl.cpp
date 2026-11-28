// OpenCL reference: runtime (CPU device, in-order queue) and arena (buffer, sub-buffers, init kernels).

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "backend_opencl/arena.hpp"
#include "backend_opencl/runtime.hpp"

namespace {

TEST(Runtime, IsCpuDevice) {
  const bench::ClRuntime rt = bench::make_runtime();
  cl_device_type type = 0;
  ASSERT_EQ(clGetDeviceInfo(rt.device.device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr), CL_SUCCESS);
  EXPECT_TRUE(type & CL_DEVICE_TYPE_CPU);
}

TEST(Runtime, QueueIsInOrderWithoutProfiling) {
  const bench::ClRuntime rt = bench::make_runtime();
  cl_command_queue_properties props = 0;
  ASSERT_EQ(clGetCommandQueueInfo(rt.queue.get(), CL_QUEUE_PROPERTIES, sizeof(props), &props, nullptr),
            CL_SUCCESS);
  EXPECT_FALSE(props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  EXPECT_FALSE(props & CL_QUEUE_PROFILING_ENABLE);
}

// Initialization incl. the init sample check (throws on a mismatch), padding and workspace
TEST(Arena, InitializeAndReadC) {
  const bench::ClRuntime rt = bench::make_runtime();
  const bench::Problem p{37, 21, 50, 53, 24, 24};
  bench::Arena arena(rt, p, 1000 * sizeof(double));
  EXPECT_NO_THROW(arena.initialize(123));

  const auto& info = arena.info();
  EXPECT_EQ(info.alloc_kind, "clCreateBuffer");
  EXPECT_EQ(info.requested_alignment, 0u);
  EXPECT_EQ(info.alignment_honored, info.base_mod_page == 0);
  EXPECT_EQ(arena.matrices().workspace_bytes, 1000 * sizeof(double));
  EXPECT_NE(arena.matrices().workspace.get(), nullptr);

  std::vector<double> c(p.m * p.ldc, 0.0);
  arena.copy_c_to_host(c.data());
  for (double v : c) ASSERT_TRUE(std::isnan(v));  // sentinel everywhere, incl. the row padding
}

TEST(Arena, NoWorkspace) {
  const bench::ClRuntime rt = bench::make_runtime();
  bench::Arena arena(rt, {5, 3, 7, 7, 3, 3}, 0);
  EXPECT_NO_THROW(arena.initialize(1));
  EXPECT_EQ(arena.matrices().workspace.get(), nullptr);
}

} // namespace
