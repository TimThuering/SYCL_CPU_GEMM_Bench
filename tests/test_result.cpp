// Result record -> JSON, status and exit codes.

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <cmath>

#include "bench/cli.hpp"
#include "bench/result.hpp"

namespace {

using nlohmann::json;

bench::ResultRecord full_record() {
  bench::ResultRecord r;
  r.config.stage = "naive";
  r.config.problem = {10, 20, 30, 30, 20, 20};
  r.config.label = "x";
  r.host = bench::query_host();
  r.env = {{"OMP_NUM_THREADS", "4"}};
  bench::TimingResult t;
  t.cold_s = 0.5;
  t.warmup_s = {0.2};
  t.times_s = {0.1, 0.3};
  bench::compute_stats(t);
  r.timing = t;
  r.os_threads = 5;
  bench::VerifyResult v;
  v.performed = true;
  v.passed = true;
  v.entries_checked = 200;
  r.verify = v;
  bench::set_completed_status(r);
  return r;
}

TEST(Result, CompleteRecord) {
  const json j = json::parse(bench::to_json(full_record()));
  EXPECT_EQ(j["schema_version"], 1);
  EXPECT_EQ(j["status"], "OK");
  EXPECT_TRUE(j["error"].is_null());
  EXPECT_EQ(j["label"], "x");
  EXPECT_EQ(j["config"]["stage"], "naive");
  EXPECT_EQ(j["config"]["k"], 30);
  EXPECT_EQ(j["env"]["OMP_NUM_THREADS"], "4");
  EXPECT_EQ(j["timing"]["times_s"].size(), 2u);
  EXPECT_DOUBLE_EQ(j["timing"]["mean_s"].get<double>(), 0.2);
  EXPECT_DOUBLE_EQ(j["timing"]["flops"].get<double>(), 2.0 * 10 * 20 * 30);
  EXPECT_EQ(j["os_threads"], 5);
  EXPECT_EQ(j["verify"]["passed"], true);
  EXPECT_TRUE(j["build"]["git"].contains("commit"));
  EXPECT_TRUE(j["device"].is_null());  // not set
  EXPECT_TRUE(j["arena"].is_null());
}

TEST(Result, VerifyFailAndNonFinite) {
  auto r = full_record();
  r.verify->passed = false;
  r.verify->max_abs_error = INFINITY;
  r.verify->failure = "C(0, 0) = nan";
  bench::set_completed_status(r);
  EXPECT_EQ(r.status, bench::Status::verify_fail);
  const json j = json::parse(bench::to_json(r));
  EXPECT_EQ(j["status"], "VERIFY_FAIL");
  EXPECT_TRUE(j["verify"]["max_abs_error"].is_null());  // inf -> null
  EXPECT_EQ(j["verify"]["failure"], "C(0, 0) = nan");
}

TEST(Result, VerificationDisabledIsOk) {
  auto r = full_record();
  r.verify = bench::VerifyResult{};  // performed = false
  bench::set_completed_status(r);
  EXPECT_EQ(r.status, bench::Status::ok);
  EXPECT_TRUE(json::parse(bench::to_json(r))["verify"]["passed"].is_null());
}

TEST(Result, ErrorRecordWithMissingParts) {
  bench::ResultRecord r;
  r.config.stage = "packed";
  r.status = bench::Status::error;
  r.error = "allocation failed";
  const json j = json::parse(bench::to_json(r));
  EXPECT_EQ(j["status"], "ERROR");
  EXPECT_EQ(j["error"], "allocation failed");
  for (const char* key : {"host", "device", "arena", "timing", "os_threads", "verify"})
    EXPECT_TRUE(j[key].is_null()) << key;
}

TEST(Result, ExitCodes) {
  EXPECT_EQ(bench::exit_code(bench::Status::ok), bench::exit_ok);
  EXPECT_EQ(bench::exit_code(bench::Status::verify_fail), bench::exit_verify_failed);
  EXPECT_EQ(bench::exit_code(bench::Status::error), bench::exit_error);
  EXPECT_EQ(bench::exit_code(bench::Status::unsupported), bench::exit_unsupported);
}

} // namespace
