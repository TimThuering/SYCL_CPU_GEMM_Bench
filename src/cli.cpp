#include "bench/cli.hpp"

#include <cxxopts.hpp>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "bench/build_info.hpp"
#include "bench/stages.hpp"

namespace bench {

namespace {

std::string stage_list_text() {
  std::string s;
  for (const auto& st : stage_names) {
    if (!s.empty()) s += ", ";
    s += st.name;
  }
  return s;
}

VerifyMode parse_verify_mode(const std::string& s) {
  if (s == "fast") return VerifyMode::fast;
  if (s == "full") return VerifyMode::full;
  if (s == "none") return VerifyMode::none;
  throw UsageError("invalid --verify '" + s + "' (expected fast | full | none)");
}

std::string join_argv(int argc, char** argv) {
  std::string out;
  for (int i = 0; i < argc; ++i) {
    std::string arg = argv[i];
    if (i > 0) out += ' ';
    if (arg.find_first_of(" \t\"'") != std::string::npos) {
      out += '\'';
      out += arg;
      out += '\'';
    } else {
      out += arg;
    }
  }
  return out;
}

// a * b, throws UsageError on size_t overflow
std::size_t mul_checked(std::size_t a, std::size_t b, const char* what) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    throw UsageError(std::string("problem too large (overflow in ") + what + ")");
  return a * b;
}

std::size_t add_checked(std::size_t a, std::size_t b, const char* what) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw UsageError(std::string("problem too large (overflow in ") + what + ")");
  return a + b;
}

void validate(Config& cfg) {
  if (cfg.stage.empty())
    throw UsageError("missing --stage (one of: " + stage_list_text() + ")");
  if (!is_stage_name(cfg.stage))
    throw UsageError("unknown stage '" + cfg.stage + "' (one of: " + stage_list_text() + ")");

  Problem& p = cfg.problem;
  if (p.m == 0 || p.n == 0 || p.k == 0)
    throw UsageError("matrix dimensions must be >= 1");

  p.lda = add_checked(p.k, cfg.ld_pad, "lda");
  p.ldb = add_checked(p.n, cfg.ld_pad, "ldb");
  p.ldc = add_checked(p.n, cfg.ld_pad, "ldc");

  // Total element count of A, B, C must fit into size_t bytes
  std::size_t elems = mul_checked(p.m, p.lda, "A");
  elems = add_checked(elems, mul_checked(p.k, p.ldb, "B"), "A+B");
  elems = add_checked(elems, mul_checked(p.m, p.ldc, "C"), "A+B+C");
  mul_checked(elems, sizeof(double), "bytes");

  if (cfg.reps == 0)
    throw UsageError("--reps must be >= 1");
  if (cfg.verify == VerifyMode::fast && cfg.verify_samples == 0)
    throw UsageError("--verify-samples must be >= 1 for --verify fast");
}

} // namespace

CliResult parse_cli(int argc, char** argv) {
  cxxopts::Options options("gemm_bench",
                           "SYCL CPU GEMM benchmark: C = A * B (double, row-major).\n"
                           "One run = one stage, one problem size; the result is written as JSON.\n");

  const Config def;  // defaults

  // clang-format off
  options.add_options("Kernel")
    ("s,stage", "Kernel stage (see --list-stages)", cxxopts::value<std::string>());

  options.add_options("Problem")
    ("size", "Sets M = N = K", cxxopts::value<std::size_t>()->default_value("1000"))
    ("m", "Rows of A and C (overrides --size)", cxxopts::value<std::size_t>())
    ("n", "Columns of B and C (overrides --size)", cxxopts::value<std::size_t>())
    ("k", "Columns of A / rows of B (overrides --size)", cxxopts::value<std::size_t>())
    ("ld-pad", "Extra elements per row (lda = K + pad, ldb = ldc = N + pad)",
        cxxopts::value<std::size_t>()->default_value(std::to_string(def.ld_pad)))
    ("seed", "Data generation seed", cxxopts::value<std::uint64_t>()->default_value(std::to_string(def.seed)));

  options.add_options("Run")
    ("warmup", "Untimed iterations after the cold one",
        cxxopts::value<std::size_t>()->default_value(std::to_string(def.warmup)))
    ("reps", "Timed iterations", cxxopts::value<std::size_t>()->default_value(std::to_string(def.reps)))
    ("profiling", "Additionally record SYCL event profiling times");

  options.add_options("Verification")
    ("verify", "Verification mode: fast | full | none",
        cxxopts::value<std::string>()->default_value(to_string(def.verify)))
    ("verify-samples", "Sampled C entries for --verify fast",
        cxxopts::value<std::size_t>()->default_value(std::to_string(def.verify_samples)));

  options.add_options("Output")
    ("o,output", "JSON output file (default: stdout)", cxxopts::value<std::string>())
    ("label", "Free-form label stored in the result", cxxopts::value<std::string>())
    ("q,quiet", "No human-readable summary on stderr");

  options.add_options("Info")
    ("h,help", "Print help")
    ("version", "Print build information")
    ("list-stages", "List kernel stages")
    ("list-devices", "List visible devices");
  // clang-format on

  CliResult res;
  res.help_text = options.help({"Kernel", "Problem", "Run", "Verification", "Output", "Info"});

  cxxopts::ParseResult r;
  try {
    r = options.parse(argc, argv);
  } catch (const cxxopts::exceptions::exception& e) {
    throw UsageError(e.what());
  }

  if (!r.unmatched().empty()) {
    std::string extra;
    for (const auto& u : r.unmatched()) extra += " " + u;
    throw UsageError("unexpected positional argument(s):" + extra);
  }

  if (r.count("help"))         { res.action = CliAction::help;         return res; }
  if (r.count("version"))      { res.action = CliAction::version;      return res; }
  if (r.count("list-stages"))  { res.action = CliAction::list_stages;  return res; }
  if (r.count("list-devices")) { res.action = CliAction::list_devices; return res; }

  Config& cfg = res.config;
  try {
    if (r.count("stage")) cfg.stage = r["stage"].as<std::string>();

    const auto size = r["size"].as<std::size_t>();
    cfg.problem.m = r.count("m") ? r["m"].as<std::size_t>() : size;
    cfg.problem.n = r.count("n") ? r["n"].as<std::size_t>() : size;
    cfg.problem.k = r.count("k") ? r["k"].as<std::size_t>() : size;
    cfg.ld_pad = r["ld-pad"].as<std::size_t>();
    cfg.seed = r["seed"].as<std::uint64_t>();

    cfg.warmup = r["warmup"].as<std::size_t>();
    cfg.reps = r["reps"].as<std::size_t>();
    cfg.profiling = r.count("profiling") > 0;

    cfg.verify = parse_verify_mode(r["verify"].as<std::string>());
    cfg.verify_samples = r["verify-samples"].as<std::size_t>();

    if (r.count("output")) cfg.output = r["output"].as<std::string>();
    if (r.count("label")) cfg.label = r["label"].as<std::string>();
    cfg.quiet = r.count("quiet") > 0;
  } catch (const cxxopts::exceptions::exception& e) {
    throw UsageError(e.what());
  }

  cfg.command_line = join_argv(argc, argv);
  validate(cfg);
  return res;
}

bool handle_common_action(const CliResult& res) {
  switch (res.action) {
    case CliAction::help:
      std::cout << res.help_text;
      return true;
    case CliAction::version:
      print_build_info(std::cout);
      return true;
    case CliAction::list_stages:
      for (const auto& st : stage_names)
        std::cout << "  " << std::left << std::setw(13) << st.name << st.description << "\n";
      return true;
    case CliAction::run:
    case CliAction::list_devices:
      return false;
  }
  return false;
}

} // namespace bench
