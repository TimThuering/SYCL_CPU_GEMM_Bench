#include "program.hpp"

#include <stdexcept>

namespace bench {

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

void enqueue_1d(const ClRuntime& rt, const ClKernel& kernel, std::size_t g0) {
  const std::size_t global[1] = {g0};
  cl_check(clEnqueueNDRangeKernel(rt.queue.get(), kernel.get(), 1, nullptr, global, nullptr, 0, nullptr,
                                  nullptr),
           "clEnqueueNDRangeKernel");
}

void enqueue_2d(const ClRuntime& rt, const ClKernel& kernel, std::size_t g0, std::size_t g1) {
  const std::size_t global[2] = {g0, g1};
  cl_check(clEnqueueNDRangeKernel(rt.queue.get(), kernel.get(), 2, nullptr, global, nullptr, 0, nullptr,
                                  nullptr),
           "clEnqueueNDRangeKernel");
}

} // namespace bench
