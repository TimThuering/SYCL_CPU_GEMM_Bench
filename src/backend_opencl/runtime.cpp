#include "runtime.hpp"

#include <stdexcept>
#include <string>

namespace bench {

ClRuntime make_runtime() {
  ClRuntime rt;
  rt.device = select_cpu_device();

  const cl_context_properties ctx_props[] = {
      CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(rt.device.platform), 0};
  cl_int err = CL_SUCCESS;
  rt.context = ClContext(clCreateContext(ctx_props, 1, &rt.device.device, nullptr, nullptr, &err));
  cl_check(err, "clCreateContext");

  // No properties: in-order execution, profiling disabled (OpenCL defaults)
  rt.queue = ClQueue(clCreateCommandQueueWithProperties(rt.context.get(), rt.device.device, nullptr, &err));
  cl_check(err, "clCreateCommandQueueWithProperties");
  return rt;
}

void finish(const ClRuntime& rt) {
  cl_check(clFinish(rt.queue.get()), "clFinish");

  // Negative execution status = the command terminated abnormally (CL_COMPLETE = 0 on success)
  std::string failure;
  for (const auto& [event, kernel] : rt.launches) {
    cl_int status = CL_COMPLETE;
    cl_check(clGetEventInfo(event.get(), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, nullptr),
             "clGetEventInfo");
    if (status < 0 && failure.empty()) {
      char name[128] = {};
      clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(name) - 1, name, nullptr);
      failure = "kernel '" + std::string(name) + "' failed to execute (execution status " +
                std::to_string(status) + ")";
    }
  }
  rt.launches.clear();
  if (!failure.empty()) throw std::runtime_error(failure);
}

} // namespace bench
