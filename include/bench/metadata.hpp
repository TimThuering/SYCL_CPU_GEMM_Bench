#pragma once
// Run-time information about the machine, the process and the device, recorded with every result.
// (Build-time information: bench/build_info.hpp.)

#include <cstddef>
#include <iosfwd>
#include <map>
#include <string>

namespace bench {

struct HostInfo {
  std::string timestamp;      // local time of the run, ISO 8601 with UTC offset
  std::string hostname;
  std::string cpu_model;      // from /proc/cpuinfo
  std::string os;             // uname: sysname release machine
  std::size_t online_cpus = 0;
  std::size_t affinity_count = 0;  // CPUs this process may run on (taskset / cgroup mask)
  std::string affinity_list;       // e.g. "0-15,32-47"
};

struct DeviceInfo {
  std::string name;
  std::string vendor;
  std::string driver_version;
  std::string device_version;
  std::string platform_name;
  std::string platform_version;
  std::size_t max_compute_units = 0;
};

HostInfo query_host();

// Environment variables that influence the SYCL implementations / OpenMP / PoCL / oneTBB
// (OMP_*, ACPP_*, POCL_*, SYCL_*, ONEAPI_*, UR_*, ...), sorted by name
std::map<std::string, std::string> relevant_env();

// Number of OS threads of this process right now (from /proc/self/status), 0 if unknown
std::size_t os_thread_count();

void print_host_info(std::ostream& os, const HostInfo& h);
void print_device_info(std::ostream& os, const DeviceInfo& d);
void print_env(std::ostream& os, const std::map<std::string, std::string>& env);

} // namespace bench
