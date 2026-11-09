// Plain C++/OpenMP twin driver (SYCL_IMPL=none).

#include <omp.h>

#include <exception>
#include <iostream>

#include "bench/build_info.hpp"
#include "bench/cli.hpp"
#include "bench/config.hpp"

namespace {

void list_devices() {
  std::cout << "  [OpenMP] host CPU, max threads = " << omp_get_max_threads()
            << " (_OPENMP = " << _OPENMP << ")\n";
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
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return bench::exit_error;
  }
}
