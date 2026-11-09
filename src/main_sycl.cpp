// SYCL driver (SYCL_IMPL=acpp|dpcpp).

#include <sycl/sycl.hpp>

#include <iostream>

#include "bench/build_info.hpp"

int main() {
  bench::print_build_info(std::cout);

  std::cout << "\nvisible SYCL devices:\n";
  for (const auto& platform : sycl::platform::get_platforms()) {
    for (const auto& dev : platform.get_devices()) {
      std::cout << "  [" << platform.get_info<sycl::info::platform::name>() << "] "
                << dev.get_info<sycl::info::device::name>() << "\n";
    }
  }

  sycl::queue q{sycl::default_selector_v};
  std::cout << "default device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  return 0;
}
