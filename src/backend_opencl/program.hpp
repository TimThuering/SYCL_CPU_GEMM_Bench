#pragma once
// OpenCL programs and kernels: build from source at run time, set arguments, launch.
//
// OpenCL C kernel sources are embedded into the binary at build time (cmake/OpenClEmbed.cmake):
// a file foo.cl becomes the string bench::cl_source::foo in the generated header cl_sources/foo.cl.hpp.

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "backend_opencl/handle.hpp"
#include "backend_opencl/runtime.hpp"

namespace bench {

// Build options of the stage programs: the compile-time kernel parameters (kernels/params.hpp)
// as macros, "-DMR=4 -DNR=8 -DKC=256 -DMC=64". No other options (no fast-math, default OpenCL C version).
std::string stage_build_options();

// Program of one stage, built on first use: the first kernel() call compiles the source
// (clBuildProgram with stage_build_options()), i.e. in the cold iteration, like the JIT compilation
// of the SYCL flows. Kernel objects are created once and reused in every iteration.
class ClStageProgram {
 public:
  ClStageProgram(const ClRuntime& rt, const char* source) : rt_(rt), source_(source) {}

  // Non-owning handle; valid as long as this ClStageProgram exists (owned by kernels_)
  cl_kernel kernel(const char* name);
  const ClRuntime& runtime() const { return rt_; }

 private:
  const ClRuntime& rt_;
  const char* source_;
  ClProgram program_;
  std::vector<std::pair<std::string, ClKernel>> kernels_;
};

// clCreateProgramWithSource + clBuildProgram for the runtime's device. On a build error, throws
// std::runtime_error with the compiler's build log.
ClProgram build_program(const ClRuntime& rt, const char* source, const std::string& options);

ClKernel make_kernel(const ClProgram& program, const char* name);

// Sets all arguments of a kernel in order (argument i = i-th value). The C++ types must match the
// kernel signature exactly: cl_mem for buffers, cl_ulong for ulong (size_t is not allowed as an
// argument type in OpenCL C).
template <typename... Args>
void set_args(cl_kernel kernel, const Args&... args) {
  cl_uint index = 0;
  (cl_check(clSetKernelArg(kernel, index++, sizeof(Args), &args), "clSetKernelArg"), ...);
}

// Enqueue over a global range without a work-group size (chosen by the implementation, like a
// SYCL range). OpenCL dimension 0 is the fastest-varying one: g0 corresponds to the rightmost
// dimension of a SYCL range. Asynchronous; all sizes must be > 0.
void enqueue_1d(const ClRuntime& rt, cl_kernel kernel, std::size_t g0);
void enqueue_2d(const ClRuntime& rt, cl_kernel kernel, std::size_t g0, std::size_t g1);

} // namespace bench
