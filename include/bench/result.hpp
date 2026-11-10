#pragma once
// Result of one run: everything recorded, written as one JSON document (schema: docs/json_schema.md).
//
// The record is filled step by step; parts that were not reached (e.g. after an error) stay empty
// and are written as null, so every run (except command line errors) produces a JSON file.

#include <cstddef>
#include <map>
#include <optional>
#include <string>

#include "bench/config.hpp"
#include "bench/errors.hpp"
#include "bench/layout.hpp"
#include "bench/metadata.hpp"
#include "bench/timing.hpp"
#include "bench/verify.hpp"

namespace bench {

inline constexpr int json_schema_version = 1;

enum class Status {
  ok,           // ran, verified (or verification disabled)
  verify_fail,  // ran, verification failed
  unsupported,  // the selected stage is not available in this build
  error,        // any other failure (allocation, SYCL exception, ...)
};

std::string to_string(Status s);
int exit_code(Status s);

struct ResultRecord {
  Status status = Status::error;
  std::string error;                          // message for unsupported / error

  Config config;
  std::optional<HostInfo> host;
  std::map<std::string, std::string> env;
  std::optional<DeviceInfo> device;
  std::optional<ArenaInfo> arena;
  std::optional<TimingResult> timing;
  std::optional<std::size_t> os_threads;      // after the timed iterations
  std::optional<VerifyResult> verify;
};

// JSON text of the record (pretty-printed)
std::string to_json(const ResultRecord& r);

// Sets the final status (ok / verify_fail) after a completed run
void set_completed_status(ResultRecord& r);

// Prints the human-readable summary (unless quiet), writes the JSON to cfg.output or stdout,
// and returns the exit code. A failure to write the JSON is reported and returns exit_error.
int finish_run(const ResultRecord& r);

} // namespace bench
