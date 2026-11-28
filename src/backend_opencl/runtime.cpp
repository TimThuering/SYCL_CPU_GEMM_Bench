#include "runtime.hpp"

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

} // namespace bench
