// SYCL driver (SYCL_IMPL=acpp|dpcpp).

#include <sycl/sycl.hpp>

#include <exception>
#include <iostream>

#include "bench/build_info.hpp"
#include "bench/cli.hpp"
#include "bench/config.hpp"

namespace {

void list_devices() {
  for (const auto& platform : sycl::platform::get_platforms()) {
    for (const auto& dev : platform.get_devices()) {
      std::cout << "  [" << platform.get_info<sycl::info::platform::name>() << "] "
                << dev.get_info<sycl::info::device::name>() << "\n";
    }
  }
  sycl::queue q{sycl::default_selector_v};
  std::cout << "default device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
}

} // namespace

int main(int argc, char** argv) {
  bench::CliResult cli;
  try {
    cli = bench::parse_cli(argc, argv);
  } catch (const bench::UsageError& e) {
    std::cerr << "error: " << e.what() << "\n(see --help)\n";
    return bench::exit_usage;
  }

  try {
    if (bench::handle_common_action(cli)) return bench::exit_ok;
    if (cli.action == bench::CliAction::list_devices) {
      list_devices();
      return bench::exit_ok;
    }

    const bench::Config& cfg = cli.config;
    if (!cfg.quiet) {
      bench::print_build_info(std::cerr);
      bench::print_config(std::cerr, cfg);
    }
    return bench::exit_ok;
  } catch (const sycl::exception& e) {
    std::cerr << "SYCL error: " << e.what() << "\n";
    return bench::exit_error;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return bench::exit_error;
  }
}
