// Plain OpenCL reference driver (SYCL_IMPL=none_opencl).

#include <exception>
#include <iostream>
#include <vector>

#include "bench/build_info.hpp"
#include "bench/cli.hpp"
#include "bench/config.hpp"
#include "bench/errors.hpp"
#include "bench/metadata.hpp"
#include "bench/result.hpp"
#include "bench/timing.hpp"
#include "bench/verify.hpp"
#include "backend_opencl/arena.hpp"
#include "backend_opencl/device.hpp"
#include "backend_opencl/gemm_args.hpp"
#include "backend_opencl/program.hpp"
#include "backend_opencl/runtime.hpp"
#include "backend_opencl/stages.hpp"

namespace {

void list_devices() {
  for (const auto& d : bench::all_devices()) {
    const bench::DeviceInfo info = bench::query_device(d);
    std::cout << "  [" << info.platform_name << "] " << info.name << " (" << info.device_version << ")\n";
  }
  const bench::ClRuntime rt = bench::make_runtime();
  const bench::DeviceInfo bench_dev = bench::query_device(rt.device);
  std::cout << "benchmark device (CPU): " << bench_dev.name << " [" << bench_dev.platform_name << "]\n";
}

// Setup and benchmark; fills the record step by step (throws on failure)
void run(bench::ResultRecord& rec) {
  const bench::Config& cfg = rec.config;

  const bench::ClStage* stage = bench::find_stage(cfg.stage);
  if (stage == nullptr)
    throw bench::UnsupportedError("stage '" + cfg.stage + "' is not registered in this build");

  // CPU device, in-order queue: kernels enqueued by one stage run one after another
  const bench::ClRuntime rt = bench::make_runtime();
  rec.device = bench::query_device(rt.device);

  bench::Arena arena(rt, cfg.problem, stage->workspace_bytes(cfg.problem), cfg.require_alignment);
  rec.arena = arena.info();
  arena.initialize(cfg.seed);
  const bench::ClGemmArgs args = bench::make_cl_gemm_args(arena.matrices(), cfg.problem);

  // Built on first use: the cold iteration includes the compilation of the stage's kernels
  bench::ClStageProgram program(rt, stage->source);

  // One iteration: enqueue the stage's work and wait for it
  rec.timing = bench::time_iterations(
      [&] {
        stage->run(program, args);
        bench::cl_check(clFinish(rt.queue.get()), "clFinish");
      },
      cfg.warmup, cfg.reps);
  rec.os_threads = bench::os_thread_count();

  if (cfg.verify) {
    std::vector<double> c(cfg.problem.m * cfg.problem.ldc);
    arena.copy_c_to_host(c.data());
    rec.verify = bench::verify(cfg, c.data());
  } else {
    rec.verify = bench::VerifyResult{};  // performed = false
  }
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

  if (bench::handle_common_action(cli)) return bench::exit_ok;
  if (cli.action == bench::CliAction::list_devices) {
    try {
      list_devices();
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return bench::exit_error;
    }
    return bench::exit_ok;
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
  } catch (const std::exception& e) {
    rec.status = bench::Status::error;
    rec.error = e.what();
  }
  return bench::finish_run(rec);
}
