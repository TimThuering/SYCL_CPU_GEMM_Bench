// SYCL driver (SYCL_IMPL=acpp|dpcpp).

#include <sycl/sycl.hpp>

#include <exception>
#include <iostream>

#include "bench/build_info.hpp"
#include "bench/cli.hpp"
#include "bench/config.hpp"
#include "bench/metadata.hpp"
#include "bench/result.hpp"
#include "backend_sycl/arena.hpp"
#include "backend_sycl/device.hpp"

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

// Setup and benchmark; fills the record step by step (throws on failure)
void run(bench::ResultRecord& rec) {
  const bench::Config& cfg = rec.config;

  sycl::queue q{sycl::default_selector_v};
  rec.device = bench::query_device(q.get_device());

  bench::Arena arena(q, cfg.problem, 0);
  rec.arena = arena.info();
  arena.initialize(cfg.seed);

  throw bench::UnsupportedError("stage '" + cfg.stage + "' is not available in this build");
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
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return bench::exit_error;
  }

  bench::ResultRecord rec;
  rec.config = cli.config;
  if (!rec.config.quiet) {
    bench::print_build_info(std::cerr);
    bench::print_config(std::cerr, rec.config);
  }

  try {
    rec.host = bench::query_host();
    rec.env = bench::relevant_env();
    run(rec);
    bench::set_completed_status(rec);
  } catch (const bench::UnsupportedError& e) {
    rec.status = bench::Status::unsupported;
    rec.error = e.what();
  } catch (const sycl::exception& e) {
    rec.status = bench::Status::error;
    rec.error = std::string("SYCL exception: ") + e.what();
  } catch (const std::exception& e) {
    rec.status = bench::Status::error;
    rec.error = e.what();
  }
  return bench::finish_run(rec);
}
