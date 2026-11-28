#pragma once
// The benchmark runtime: CPU device (see device.hpp), its context and one in-order command queue
// without profiling (counterpart of the SYCL in-order queue in backend_sycl/queue.hpp).

#include "backend_opencl/device.hpp"
#include "backend_opencl/handle.hpp"

namespace bench {

struct ClRuntime {
  ClDevice device;
  ClContext context;  // declared before the queue: the queue is released first
  ClQueue queue;
};

// Throws std::runtime_error if no CPU device is visible or a creation call fails
ClRuntime make_runtime();

} // namespace bench
