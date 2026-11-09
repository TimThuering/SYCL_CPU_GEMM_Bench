#pragma once
// Build information embedded at configure/build time (see cmake/BuildInfo.cmake).

#include <cstddef>
#include <iosfwd>
#include <string>

namespace bench {

struct BuildInfo {
  std::string sycl_impl;        // acpp | dpcpp | none
  std::string sycl_targets;     // --acpp-targets / -fsycl-targets value
  std::string sycl_extra_flags;
  std::string arch_flags;
  std::string effective_flags;  // SYCL + extra + arch flags as passed to the compiler
  std::string cxx_flags;        // CMAKE_CXX_FLAGS + build type flags
  std::string build_type;
  std::string compiler_path;
  std::string compiler_id;
  std::string compiler_version;
  std::string compiler_version_output;  // full `<compiler> --version`
  std::string cmake_version;
  std::string build_host;       // build machine (may differ from the machine running the binary)
  std::string build_cpu;
  std::string build_cores;      // logical cores of the build machine
  std::string build_os;
  std::string build_arch;
  std::string git_commit;
  std::string git_describe;
  std::string git_dirty;        // "true" | "false" | "unknown"
  std::size_t mr, nr, kc, mc;   // kernel parameters (kernels/params.hpp)
};

const BuildInfo& build_info();

// Human-readable summary
void print_build_info(std::ostream& os);

} // namespace bench
