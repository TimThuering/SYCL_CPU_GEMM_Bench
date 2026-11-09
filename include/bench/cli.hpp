#pragma once
// Command line parsing (cxxopts) and validation.

#include <stdexcept>
#include <string>

#include "bench/config.hpp"

namespace bench {

// Invalid command line (exit code 2)
struct UsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class CliAction {
  run,           // run the benchmark with the parsed Config
  help,          // print help_text
  version,       // print build info
  list_stages,   // print stage names
  list_devices,  // print visible devices (driver specific)
};

struct CliResult {
  CliAction action = CliAction::run;
  Config config;          // valid if action == run
  std::string help_text;
};

// Parses and validates argv. Throws UsageError on invalid input.
CliResult parse_cli(int argc, char** argv);

// Handles help / version / list_stages (printed to stdout).
// Returns false for run and list_devices, which the driver handles.
bool handle_common_action(const CliResult& res);

// Exit codes
inline constexpr int exit_ok = 0;
inline constexpr int exit_verify_failed = 1;
inline constexpr int exit_usage = 2;
inline constexpr int exit_error = 3;

} // namespace bench
