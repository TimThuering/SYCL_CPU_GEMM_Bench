#pragma once
// OpenCL platforms and devices: listing, selection of the benchmark device, device information.
//
// The benchmark device is the first CPU device of the first platform that has one (never a GPU,
// even if one is visible). Which platforms are visible is controlled through the ICD loader
// (OCL_ICD_VENDORS / OCL_ICD_FILENAMES), the PoCL driver through POCL_DEVICES.

#include <vector>

#include "backend_opencl/cl_check.hpp"
#include "bench/metadata.hpp"

namespace bench {

struct ClDevice {
  cl_platform_id platform;
  cl_device_id device;
};

// All devices of all platforms (empty if no platform is installed)
std::vector<ClDevice> all_devices();

// Throws std::runtime_error if no CPU device is visible
ClDevice select_cpu_device();

DeviceInfo query_device(const ClDevice& d);

} // namespace bench
