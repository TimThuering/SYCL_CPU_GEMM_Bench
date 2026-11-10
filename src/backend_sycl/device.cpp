#include "device.hpp"

namespace bench {

DeviceInfo query_device(const sycl::device& dev) {
  const sycl::platform plat = dev.get_platform();
  DeviceInfo d;
  d.name = dev.get_info<sycl::info::device::name>();
  d.vendor = dev.get_info<sycl::info::device::vendor>();
  d.driver_version = dev.get_info<sycl::info::device::driver_version>();
  d.device_version = dev.get_info<sycl::info::device::version>();
  d.platform_name = plat.get_info<sycl::info::platform::name>();
  d.platform_version = plat.get_info<sycl::info::platform::version>();
  d.max_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
  return d;
}

} // namespace bench
