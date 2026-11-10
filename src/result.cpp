#include "bench/result.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>

#include "bench/build_info.hpp"
#include "bench/cli.hpp"

namespace bench {

using nlohmann::ordered_json;  // keeps the insertion order of the keys in the output

namespace {

// Decimal string -> number, null if it is not a plain number (e.g. "unknown")
ordered_json number_or_null(const std::string& s) {
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) return nullptr;
  return std::stoull(s);
}

// JSON has no inf / NaN: non-finite numbers are written as null
ordered_json number(double x) {
  return std::isfinite(x) ? ordered_json(x) : ordered_json(nullptr);
}

// "true" / "false" / anything else -> true / false / null
ordered_json tri_bool(const std::string& s) {
  if (s == "true") return true;
  if (s == "false") return false;
  return nullptr;
}

template <class T, class F>
ordered_json optional_json(const std::optional<T>& v, F&& to) {
  return v ? to(*v) : ordered_json(nullptr);
}

ordered_json build_json() {
  const BuildInfo& b = build_info();
  ordered_json j;
  j["sycl_impl"] = b.sycl_impl;
  j["sycl_targets"] = b.sycl_targets;
  j["sycl_extra_flags"] = b.sycl_extra_flags;
  j["arch_flags"] = b.arch_flags;
  j["effective_flags"] = b.effective_flags;
  j["cxx_flags"] = b.cxx_flags;
  j["build_type"] = b.build_type;
  j["compiler"] = {{"path", b.compiler_path},
                   {"id", b.compiler_id},
                   {"version", b.compiler_version},
                   {"version_output", b.compiler_version_output}};
  j["cmake_version"] = b.cmake_version;
  j["git"] = {{"commit", b.git_commit},
              {"describe", b.git_describe},
              {"dirty", tri_bool(b.git_dirty)},
              {"source", b.git_source}};
  j["build_host"] = {{"hostname", b.build_host},
                     {"cpu", b.build_cpu},
                     {"logical_cores", number_or_null(b.build_cores)},
                     {"os", b.build_os},
                     {"arch", b.build_arch}};
  j["params"] = {{"MR", b.mr}, {"NR", b.nr}, {"KC", b.kc}, {"MC", b.mc}};
  return j;
}

ordered_json config_json(const Config& c) {
  const Problem& p = c.problem;
  ordered_json j;
  j["stage"] = c.stage;
  j["m"] = p.m;
  j["n"] = p.n;
  j["k"] = p.k;
  j["lda"] = p.lda;
  j["ldb"] = p.ldb;
  j["ldc"] = p.ldc;
  j["ld_pad"] = c.ld_pad;
  j["seed"] = c.seed;
  j["warmup"] = c.warmup;
  j["reps"] = c.reps;
  j["verify"] = c.verify;
  return j;
}

ordered_json host_json(const HostInfo& h) {
  ordered_json j;
  j["hostname"] = h.hostname;
  j["cpu_model"] = h.cpu_model;
  j["os"] = h.os;
  j["online_cpus"] = h.online_cpus;
  j["affinity"] = {{"count", h.affinity_count}, {"list", h.affinity_list}};
  return j;
}

ordered_json device_json(const DeviceInfo& d) {
  ordered_json j;
  j["name"] = d.name;
  j["vendor"] = d.vendor;
  j["driver_version"] = d.driver_version;
  j["device_version"] = d.device_version;
  j["platform_name"] = d.platform_name;
  j["platform_version"] = d.platform_version;
  j["max_compute_units"] = d.max_compute_units;
  return j;
}

ordered_json arena_json(const ArenaInfo& a) {
  const ArenaLayout& l = a.layout;
  ordered_json j;
  j["alloc_kind"] = a.alloc_kind;
  j["alignment"] = a.alignment;
  j["total_bytes"] = l.total_bytes;
  j["regions"] = {{"A", {{"offset", l.a_offset}, {"bytes", l.a_bytes}}},
                  {"B", {{"offset", l.b_offset}, {"bytes", l.b_bytes}}},
                  {"C", {{"offset", l.c_offset}, {"bytes", l.c_bytes}}},
                  {"workspace", {{"offset", l.ws_offset}, {"bytes", l.ws_bytes}}}};
  return j;
}

ordered_json timing_json(const TimingResult& t, double flops) {
  ordered_json j;
  j["clock"] = "std::chrono::steady_clock";
  j["flops"] = flops;
  j["cold_s"] = t.cold_s;
  j["warmup_s"] = t.warmup_s;
  j["times_s"] = t.times_s;
  j["mean_s"] = t.mean_s;
  j["min_s"] = t.min_s;
  j["max_s"] = t.max_s;
  j["gflops_mean"] = number(gflops(flops, t.mean_s));
  j["gflops_best"] = number(gflops(flops, t.min_s));
  return j;
}

ordered_json verify_json(const VerifyResult& v) {
  ordered_json j;
  j["performed"] = v.performed;
  j["passed"] = v.performed ? ordered_json(v.passed) : ordered_json(nullptr);
  j["entries_checked"] = v.entries_checked;
  j["max_abs_error"] = number(v.max_abs_error);
  j["tolerance"] = verify_tolerance;
  j["failure"] = v.failure.empty() ? ordered_json(nullptr) : ordered_json(v.failure);
  return j;
}

void print_summary(std::ostream& os, const ResultRecord& r) {
  if (r.host) print_host_info(os, *r.host);
  print_env(os, r.env);
  if (r.device) print_device_info(os, *r.device);
  if (r.arena) print_arena_info(os, *r.arena);
  if (r.timing) print_timing(os, *r.timing, r.config.problem.flops());
  if (r.os_threads) os << "OS threads:    " << *r.os_threads << "\n";
  if (r.verify) print_verify_result(os, *r.verify);
  os << "status:        " << to_string(r.status);
  if (!r.error.empty()) os << " (" << r.error << ")";
  os << "\n";
}

} // namespace

std::string to_string(Status s) {
  switch (s) {
    case Status::ok: return "OK";
    case Status::verify_fail: return "VERIFY_FAIL";
    case Status::unsupported: return "UNSUPPORTED";
    case Status::error: return "ERROR";
  }
  return "ERROR";
}

int exit_code(Status s) {
  switch (s) {
    case Status::ok: return exit_ok;
    case Status::verify_fail: return exit_verify_failed;
    case Status::unsupported: return exit_unsupported;
    case Status::error: return exit_error;
  }
  return exit_error;
}

void set_completed_status(ResultRecord& r) {
  r.status = (r.verify && r.verify->performed && !r.verify->passed) ? Status::verify_fail : Status::ok;
}

std::string to_json(const ResultRecord& r) {
  ordered_json j;
  j["schema_version"] = json_schema_version;
  j["status"] = to_string(r.status);
  j["error"] = r.error.empty() ? ordered_json(nullptr) : ordered_json(r.error);
  j["label"] = r.config.label;
  j["timestamp"] = r.host ? ordered_json(r.host->timestamp) : ordered_json(nullptr);
  j["command_line"] = r.config.command_line;
  j["config"] = config_json(r.config);
  j["build"] = build_json();
  j["host"] = optional_json(r.host, host_json);
  j["env"] = r.env;
  j["device"] = optional_json(r.device, device_json);
  j["arena"] = optional_json(r.arena, arena_json);
  j["timing"] = r.timing ? timing_json(*r.timing, r.config.problem.flops()) : ordered_json(nullptr);
  j["os_threads"] = r.os_threads ? ordered_json(*r.os_threads) : ordered_json(nullptr);
  j["verify"] = optional_json(r.verify, verify_json);
  return j.dump(2);
}

int finish_run(const ResultRecord& r) {
  if (!r.config.quiet) print_summary(std::cerr, r);

  const std::string text = to_json(r) + "\n";
  if (r.config.output.empty()) {
    std::cout << text << std::flush;
  } else {
    std::ofstream f(r.config.output);
    f << text;
    f.close();
    if (!f) {
      std::cerr << "error: could not write the result to '" << r.config.output << "'\n";
      return exit_error;
    }
  }
  return exit_code(r.status);
}

} // namespace bench
