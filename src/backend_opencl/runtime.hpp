#pragma once
// The benchmark runtime: CPU device (see device.hpp), its context and one in-order command queue
// without profiling (counterpart of the SYCL in-order queue in backend_sycl/queue.hpp).
//
// Error reporting: clFinish only reports errors of the queue itself, not that an enqueued kernel
// failed to execute (e.g. the driver could not load the compiled kernel). Every kernel launch
// therefore records its event, and finish() checks the execution status of all of them
// (counterpart of SYCL's wait_and_throw for asynchronous errors).

#include <utility>
#include <vector>

#include "backend_opencl/device.hpp"
#include "backend_opencl/handle.hpp"

namespace bench {

struct ClRuntime {
  ClDevice device;
  ClContext context;  // declared before the queue: the queue is released first
  ClQueue queue;
  // Events of the kernels enqueued since the last finish() (with the kernel, for the error message);
  // declared last: released before queue and context
  mutable std::vector<std::pair<ClEvent, cl_kernel>> launches;
};

// Throws std::runtime_error if no CPU device is visible or a creation call fails
ClRuntime make_runtime();

// Waits for all enqueued commands (clFinish), then checks the execution status of every kernel
// launched since the last call; throws std::runtime_error naming the first failed kernel.
void finish(const ClRuntime& rt);

} // namespace bench
