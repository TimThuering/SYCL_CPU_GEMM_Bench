#include "device.hpp"

#include <stdexcept>
#include <string>

namespace bench {

namespace {

std::string platform_string(cl_platform_id p, cl_platform_info what) {
  std::size_t size = 0;
  cl_check(clGetPlatformInfo(p, what, 0, nullptr, &size), "clGetPlatformInfo");
  std::string s(size, '\0');
  cl_check(clGetPlatformInfo(p, what, size, s.data(), nullptr), "clGetPlatformInfo");
  if (!s.empty() && s.back() == '\0') s.pop_back();  // size includes the terminating NUL
  return s;
}

std::string device_string(cl_device_id d, cl_device_info what) {
  std::size_t size = 0;
  cl_check(clGetDeviceInfo(d, what, 0, nullptr, &size), "clGetDeviceInfo");
  std::string s(size, '\0');
  cl_check(clGetDeviceInfo(d, what, size, s.data(), nullptr), "clGetDeviceInfo");
  if (!s.empty() && s.back() == '\0') s.pop_back();
  return s;
}

std::vector<cl_platform_id> platforms() {
  cl_uint count = 0;
  const cl_int err = clGetPlatformIDs(0, nullptr, &count);
  // the ICD loader reports "no platform installed" as an error
  if (err == CL_PLATFORM_NOT_FOUND_KHR) return {};
  cl_check(err, "clGetPlatformIDs");
  std::vector<cl_platform_id> ids(count);
  if (count > 0) cl_check(clGetPlatformIDs(count, ids.data(), nullptr), "clGetPlatformIDs");
  return ids;
}

std::vector<cl_device_id> devices(cl_platform_id p, cl_device_type type) {
  cl_uint count = 0;
  const cl_int err = clGetDeviceIDs(p, type, 0, nullptr, &count);
  if (err == CL_DEVICE_NOT_FOUND) return {};
  cl_check(err, "clGetDeviceIDs");
  std::vector<cl_device_id> ids(count);
  if (count > 0) cl_check(clGetDeviceIDs(p, type, count, ids.data(), nullptr), "clGetDeviceIDs");
  return ids;
}

} // namespace

std::vector<ClDevice> all_devices() {
  std::vector<ClDevice> result;
  for (cl_platform_id p : platforms())
    for (cl_device_id d : devices(p, CL_DEVICE_TYPE_ALL)) result.push_back({p, d});
  return result;
}

ClDevice select_cpu_device() {
  for (cl_platform_id p : platforms()) {
    const auto cpus = devices(p, CL_DEVICE_TYPE_CPU);
    if (!cpus.empty()) return {p, cpus.front()};
  }
  throw std::runtime_error("no OpenCL CPU device found (check OCL_ICD_VENDORS / POCL_DEVICES)");
}

DeviceInfo query_device(const ClDevice& d) {
  DeviceInfo info;
  info.name = device_string(d.device, CL_DEVICE_NAME);
  info.vendor = device_string(d.device, CL_DEVICE_VENDOR);
  info.driver_version = device_string(d.device, CL_DRIVER_VERSION);
  info.device_version = device_string(d.device, CL_DEVICE_VERSION);
  info.platform_name = platform_string(d.platform, CL_PLATFORM_NAME);
  info.platform_version = platform_string(d.platform, CL_PLATFORM_VERSION);
  cl_uint units = 0;
  cl_check(clGetDeviceInfo(d.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, nullptr),
           "clGetDeviceInfo");
  info.max_compute_units = units;
  return info;
}

} // namespace bench
