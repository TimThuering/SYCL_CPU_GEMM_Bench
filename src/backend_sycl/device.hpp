#pragma once
// Device information of the SYCL queue's device.

#include <sycl/sycl.hpp>

#include "bench/metadata.hpp"

namespace bench {

DeviceInfo query_device(const sycl::device& dev);

} // namespace bench
