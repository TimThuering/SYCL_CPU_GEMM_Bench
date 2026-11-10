#include "device.hpp"

#include <omp.h>

#include <string>

namespace bench {

DeviceInfo query_device() {
  DeviceInfo d;
  d.name = "host CPU (OpenMP)";
  d.driver_version = "_OPENMP " + std::to_string(_OPENMP);
  d.platform_name = "OpenMP";
  d.max_compute_units = static_cast<std::size_t>(omp_get_max_threads());  // threads of a parallel region
  return d;
}

} // namespace bench
