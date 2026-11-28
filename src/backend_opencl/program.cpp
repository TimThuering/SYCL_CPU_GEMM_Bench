#include "program.hpp"

#include <stdexcept>

#include "params.hpp"

namespace bench {

std::string stage_build_options() {
  return "-DMR=" + std::to_string(gemm::MR) + " -DNR=" + std::to_string(gemm::NR) +
         " -DKC=" + std::to_string(gemm::KC) + " -DMC=" + std::to_string(gemm::MC);
}

cl_kernel ClStageProgram::kernel(const char* name) {
  if (program_.get() == nullptr) program_ = build_program(rt_, source_, stage_build_options());
  for (const auto& [n, k] : kernels_)
    if (n == name) return k.get();
  // When kernels_ grows, the ClKernel owners are moved, but the cl_kernel values they hold do not
  // change: handles returned earlier stay valid
  kernels_.emplace_back(name, make_kernel(program_, name));
  return kernels_.back().second.get();
}

ClProgram build_program(const ClRuntime& rt, const char* source, const std::string& options) {
  cl_int err = CL_SUCCESS;
  ClProgram program(clCreateProgramWithSource(rt.context.get(), 1, &source, nullptr, &err));
  cl_check(err, "clCreateProgramWithSource");

  const cl_device_id dev = rt.device.device;
  err = clBuildProgram(program.get(), 1, &dev, options.c_str(), nullptr, nullptr);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    std::size_t size = 0;
    clGetProgramBuildInfo(program.get(), dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
    std::string log(size, '\0');
    clGetProgramBuildInfo(program.get(), dev, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
    throw std::runtime_error("clBuildProgram failed (options '" + options + "'):\n" + log);
  }
  cl_check(err, "clBuildProgram");
  return program;
}

ClKernel make_kernel(const ClProgram& program, const char* name) {
  cl_int err = CL_SUCCESS;
  ClKernel kernel(clCreateKernel(program.get(), name, &err));
  cl_check(err, ("clCreateKernel(" + std::string(name) + ")").c_str());
  return kernel;
}

void enqueue_1d(const ClRuntime& rt, cl_kernel kernel, std::size_t g0) {
  const std::size_t global[1] = {g0};
  cl_check(clEnqueueNDRangeKernel(rt.queue.get(), kernel, 1, nullptr, global, nullptr, 0, nullptr, nullptr),
           "clEnqueueNDRangeKernel");
}

void enqueue_2d(const ClRuntime& rt, cl_kernel kernel, std::size_t g0, std::size_t g1) {
  const std::size_t global[2] = {g0, g1};
  cl_check(clEnqueueNDRangeKernel(rt.queue.get(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
           "clEnqueueNDRangeKernel");
}

} // namespace bench
