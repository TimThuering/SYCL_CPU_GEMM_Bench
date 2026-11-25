#pragma once
// The benchmark queue: a CPU device (never a GPU, even if one is visible), in-order.
// Which CPU device (OpenMP host device, PoCL, Native CPU) is chosen through the environment
// (ACPP_VISIBILITY_MASK, ONEAPI_DEVICE_SELECTOR). Throws sycl::exception if no CPU device is visible.

#include <sycl/sycl.hpp>

namespace bench {

inline sycl::queue make_queue() {
  return sycl::queue{sycl::cpu_selector_v, sycl::property::queue::in_order{}};
}

} // namespace bench
